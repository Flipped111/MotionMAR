import math
from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from VQVAE.transformer_vqvae import TransformerVQVAE as VQVAE
from VQVAE.quantize import VectorQuantizer2


class SparseConditionEncoder(nn.Module):
    """Encodes sparse tracking input into multi-scale conditional features."""
    def __init__(self, input_dim=18*3, embed_dim=1024, scales=(5,10,20)):
        super().__init__()
        self.scales = scales
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.body = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 3, 1, 1),
        )

    def forward(self, sparse_seq):
        """
        sparse_seq: [B, T, J, 18] flattened to [B, T, sparse_input_dim].
        Returns: features per patch_num concatenated as [B, sum(scales), Dim].
        """
        B, T, = sparse_seq.shape[0:2]
        x = sparse_seq.reshape(B, T, -1)

        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.body(x)

        # Resample to each VAR scale via 1D linear interpolation.
        multi_scale_feats = []
        for s in self.scales:
            feat = F.interpolate(x, size=s, mode='linear', align_corners=False)
            multi_scale_feats.append(feat.permute(0, 2, 1))

        # Concat in the same scale order as VAR token layout.
        return torch.cat(multi_scale_feats, dim=1)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(5,10,20),
        flash_if_available=True, fused_if_available=True,
        sparse_input_dim=54,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.e_dim, vae_local.n_e
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1

        self.patch_nums: Tuple[int] = patch_nums
        # Motion patch_nums are 1D temporal lengths, so total length L = sum(pn).
        self.L = sum(pn for pn in self.patch_nums)
        self.first_l = self.patch_nums[0]

        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn))
            cur += pn

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantizer
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. Condition embedding: learnable SOS token + sparse condition encoder.
        init_std = math.sqrt(1 / self.C / 3)
        self.sos_token = nn.Parameter(torch.empty(1, 1, self.C))
        nn.init.trunc_normal_(self.sos_token.data, mean=0, std=init_std)

        self.cond_encoder = SparseConditionEncoder(
            input_dim=sparse_input_dim,
            embed_dim=self.C,
            scales=self.patch_nums
        )

        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. 1D absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        self.pos_1LC = nn.Parameter(pos_1LC)

        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        # 5. attention mask (1D)
        d: torch.Tensor = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual, cond_BD):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def _logits_to_latent(self, logits_BlV, decode, tau, soft_top_k):
        """Convert token logits to codebook embeddings for the next scale."""
        embed_w = self.vae_quant_proxy[0].embedding.weight
        if decode == "soft":
            if soft_top_k > 0:
                values, indices = logits_BlV.topk(soft_top_k, dim=-1)
                probs = torch.softmax(values / tau, dim=-1)
                h_BlC = (probs.unsqueeze(-1) * embed_w[indices]).sum(dim=2)
            else:
                probs = torch.softmax(logits_BlV / tau, dim=-1)
                h_BlC = probs @ embed_w
        elif decode == "sample":
            batch, length, vocab = logits_BlV.shape
            if soft_top_k > 0:
                values, indices = logits_BlV.topk(soft_top_k, dim=-1)
                probs = torch.softmax(values / tau, dim=-1).reshape(-1, soft_top_k)
                choices = torch.multinomial(probs, 1).reshape(batch, length, 1)
                idx_Bl = torch.gather(indices, -1, choices).squeeze(-1)
            else:
                probs = torch.softmax(logits_BlV / tau, dim=-1).reshape(-1, vocab)
                idx_Bl = torch.multinomial(probs, 1).reshape(batch, length)
            h_BlC = self.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            idx_Bl = torch.argmax(logits_BlV, dim=-1)
            h_BlC = self.vae_quant_proxy[0].embedding(idx_Bl)
        return h_BlC.transpose(1, 2)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, sparse_seq: torch.Tensor,
        g_seed: Optional[int] = None, cfg_scale: float = 1.0,
        decode: str = "argmax", tau: float = 1.0, soft_top_k: int = 0,
    ) -> torch.Tensor:
        """
        sparse_seq: [B, T_max, J, 18]
        """
        if decode not in ("argmax", "soft", "sample"):
            raise ValueError(f"Unsupported decode mode: {decode}")
        if tau <= 0:
            raise ValueError("tau must be greater than 0.")
        if soft_top_k < 0 or soft_top_k > self.V:
            raise ValueError(f"soft_top_k must be between 0 and {self.V}.")
        if g_seed is not None:
            self.rng.manual_seed(g_seed)

        use_cfg = abs(cfg_scale - 1.0) > 1e-6

        # 1. Encode condition: cond_feats_all is [B, L_total, C].
        cond_feats_all = self.cond_encoder(sparse_seq)
        cond_s0 = cond_feats_all[:, :self.first_l, :]

        # 2. SOS token: [B, 1, C]
        sos = self.sos_token.expand(B, 1, -1)

        # 3. AdaLN global condition: SOS + mean-pooled sparse features.
        sos_BD = sos.squeeze(1)
        cond_BD_c = sos_BD + cond_feats_all.mean(dim=1)

        # 4. Initialise scale-0 token map by injecting sparse condition.
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        base0 = sos.expand(B, self.first_l, -1) + \
                self.pos_start.expand(B, self.first_l, -1) + \
                lvl_pos[:, :self.first_l]
        next_token_map_c = base0 + cond_s0

        if use_cfg:
            cond_BD = torch.cat([cond_BD_c, sos_BD], dim=0)
            next_token_map = torch.cat([next_token_map_c, base0], dim=0)
        else:
            cond_BD = cond_BD_c
            next_token_map = next_token_map_c

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            cur_L += pn

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map

            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            if use_cfg:
                logits_BlV = (cfg_scale * logits_BlV[:B] -
                              (cfg_scale - 1.0) * logits_BlV[B:])

            h_BCt = self._logits_to_latent(logits_BlV, decode, tau, soft_top_k)

            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BCt)

            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.transpose(1, 2)
                next_token_map = self.word_embed(next_token_map)

                # Add pos + level embedding + next-scale condition features.
                next_len = self.patch_nums[si+1]
                cond_next = cond_feats_all[:, cur_L : cur_L + next_len, :]

                base = next_token_map + lvl_pos[:, cur_L : cur_L + next_len]
                if use_cfg:
                    next_token_map = torch.cat([base + cond_next, base], dim=0)
                else:
                    next_token_map = base + cond_next

        for b in self.blocks: b.attn.kv_caching(False)

        return self.vae_proxy[0].decode_my(f_hat, sparse_seq)


    def forward(self, sparse_seq: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor):
        """
        sparse_seq: [B, T, J, 18] (Condition)
        x_BLCv_wo_first_l: [B, L_remain, Cvae] (Teacher Forcing Input)
        """
        B = x_BLCv_wo_first_l.shape[0]
        cond_feats_all = self.cond_encoder(sparse_seq)

        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)

        with torch.amp.autocast("cuda",enabled=False):
            # Random condition drop for CFG training.
            if self.training and self.cond_drop_rate > 0:
                mask = torch.rand(B, device=sparse_seq.device) < self.cond_drop_rate
                # Clone before zeroing so the original tensor is preserved for downstream use.
                cond_feats_masked = cond_feats_all.clone()
                cond_feats_masked[mask] = 0

                cond_BD = self.sos_token.squeeze(1).expand(B, -1) + cond_feats_masked.mean(dim=1)
                seq_cond = cond_feats_masked
            else:
                cond_BD = self.sos_token.squeeze(1).expand(B, -1) + cond_feats_all.mean(dim=1)
                seq_cond = cond_feats_all

            sos = self.sos_token.expand(B, self.first_l, -1) + \
                  self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                # Concat teacher-forcing VQ embeddings.
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            # Add positional, level, and sparse-conditioning embeddings (truncated to ed).
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + \
                     self.pos_1LC[:, :ed] + \
                     seq_cond[:, :ed, :]

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # Cast Types
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s

        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'



if __name__=="__main__":
    model = VAR(VQVAE().cuda(1)).cuda(1)
    data1=torch.Tensor(2,20,3,18).cuda(1)
    out=model.autoregressive_infer_cfg(2,data1)
    print(out.shape)
