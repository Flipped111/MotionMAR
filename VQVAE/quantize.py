from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

__all__ = ['VectorQuantizer2',]

class Phi1D(nn.Conv1d):
    """1D residual convolution used to mix features across scales."""
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCT):
        return h_BCT.mul(1-self.resi_ratio) + super().forward(h_BCT).mul_(self.resi_ratio)

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi1D):
        super().__init__()
        self.qresi: Phi1D = qresi
    def __getitem__(self, _) -> Phi1D:
        return self.qresi

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    def __getitem__(self, at_from_0_to_1: float) -> Phi1D:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    def __getitem__(self, at_from_0_to_1: float) -> Phi1D:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

class VectorQuantizer2(nn.Module):
    """VAR quantizer for (B, C, T) motion latents.

    v_patch_nums encodes the temporal scales used by VAR.
    """
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums  # e.g., (8, 16, 32, 64)

        self.quant_resi_ratio = quant_resi

        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared([(Phi1D(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(Phi1D(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi1D(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        self.prog_si = -1

    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def forward(self, f_BCT: torch.Tensor, ret_usages=True) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        f_BCT: (Batch, Channel, Time) - Motion Latent Features
        """
        dtype = f_BCT.dtype
        if dtype != torch.float32: f_BCT = f_BCT.float()
        B, C, T = f_BCT.shape
        f_no_grad = f_BCT.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast("cuda",enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCT.device)
            SN = len(self.v_patch_nums)
            
            for si, tn in enumerate(self.v_patch_nums): # tn: temporal number (length)
                # 1. Downsample (1D linear interpolation works well for temporal signals).
                if si != SN-1:
                    # (B, C, T) -> (B, C, tn) -> (B, tn, C) -> (B*tn, C)
                    rest_NC = F.interpolate(f_rest, size=tn, mode='linear', align_corners=False).permute(0, 2, 1).reshape(-1, C)
                else:
                    rest_NC = f_rest.permute(0, 2, 1).reshape(-1, C)
                
                # 2. Find Nearest Embedding
                if self.using_znorm:
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()

                # 3. Restore / Upsample
                idx_Bt = idx_N.view(B, tn) # (B, t_scale)
                
                # Embed: (B, t, C) -> (B, C, t)
                h_BCt = self.embedding(idx_Bt).permute(0, 2, 1)
                
                # Upsample back to original T (Input Length)
                if si != SN-1:
                    h_BCT = F.interpolate(h_BCt, size=T, mode='linear', align_corners=False).contiguous()
                else:
                    h_BCT = h_BCt.contiguous()
                
                # 4. Residual Processing (Phi1D)
                h_BCT = self.quant_resi[si/(SN-1)](h_BCT)
                
                f_hat = f_hat + h_BCT
                f_rest -= h_BCT
                
                if self.training:
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                
                # Loss accumulation
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCT).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCT)
            
        margin = 1 * (f_BCT.numel() / f_BCT.shape[1]) / self.vocab_size * 0.08
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, tn in enumerate(self.v_patch_nums)]
        else: usages = None
        return f_hat, usages, mean_vq_loss

    def f_to_idxBl_or_fhat(self, f_BCT: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[int]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """Inference helper: returns multi-scale token sequences or partial f_hat."""
        B, C, T = f_BCT.shape
        f_no_grad = f_BCT.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        # Scales: list of time steps [t1, t2, ..., T]
        temp_lens = v_patch_nums or self.v_patch_nums
        assert temp_lens[-1] == T, f'{temp_lens[-1]=} != {T=}'
        
        SN = len(temp_lens)
        for si, tn in enumerate(temp_lens):
            if 0 <= self.prog_si < si: break
            
            # Downsample & Find Index
            z_NC = F.interpolate(f_rest, size=tn, mode='linear', align_corners=False).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
            
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            # Reconstruct for residual calculation
            idx_Bt = idx_N.view(B, tn)
            h_BCT = F.interpolate(self.embedding(idx_Bt).permute(0, 2, 1), size=T, mode='linear', align_corners=False).contiguous() if (si != SN-1) else self.embedding(idx_Bt).permute(0, 2, 1).contiguous()
            
            h_BCT = self.quant_resi[si/(SN-1)](h_BCT)
            
            f_hat.add_(h_BCT)
            f_rest.sub_(h_BCT)
            
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, tn))
        
        return f_hat_or_idx_Bl

    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Build teacher-forcing input for the VAR Transformer.

        Embeds tokens at each scale, accumulates them into f_hat, and resamples to
        the next scale, concatenating along the time dimension.
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        T = self.v_patch_nums[-1] # Max Length
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, T, dtype=torch.float32)
        tn_next: int = self.v_patch_nums[0]
        
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break

            curr_idx = gt_ms_idx_Bl[si] # (B, t_curr)
            h_BCt = self.embedding(curr_idx).transpose(1, 2) # (B, C, t_curr)
            
            # Upsample to Max T to apply Residual Phi
            h_BCT = F.interpolate(h_BCt, size=T, mode='linear', align_corners=False)
            h_BCT = self.quant_resi[si/(SN-1)](h_BCT)
            f_hat.add_(h_BCT)
            
            # Downsample to NEXT scale to get the condition for next step
            tn_next = self.v_patch_nums[si+1]
            next_scale_feat = F.interpolate(f_hat, size=tn_next, mode='linear', align_corners=False) # (B, C, t_next)
            
            # Transpose to (B, t_next, C) for transformer
            next_scales.append(next_scale_feat.transpose(1, 2))
            
        return torch.cat(next_scales, dim=1) if len(next_scales) else None # Concat along Time dimension

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BCT: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Used by VAR during inference to advance to the next scale."""
        T = self.v_patch_nums[-1]
        if si != SN-1:
            # Current scale residual
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BCT, size=T, mode='linear', align_corners=False))
            f_hat.add_(h)
            # Resize entire f_hat to next scale size
            return f_hat, F.interpolate(f_hat, size=self.v_patch_nums[si+1], mode='linear', align_corners=False)
        else:
            h = self.quant_resi[si/(SN-1)](h_BCT)
            f_hat.add_(h)
            return f_hat, f_hat

if __name__=="__main__":
    scales = (5,10,20)
    model = VectorQuantizer2(vocab_size=1024, Cvae=256, using_znorm=True, v_patch_nums=scales)
    model.cuda()
    
    dummy_input = torch.randn(2, 256, 20).cuda()
    
    f_hat, usages, loss = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {f_hat.shape}")
    print(f"Loss: {loss.item()}")
    print(usages)
