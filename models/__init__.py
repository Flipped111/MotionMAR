from typing import Tuple
import torch.nn as nn

from .var import VAR
from VQVAE.transformer_vqvae import TransformerVQVAE as VQVAE


def build_vae_var(
    device, patch_nums=(5, 10, 20),   # 3 scales for motion
    V=1024, Cvae=64, share_quant_resi=4,
    vqvae_in_dim=132, vqvae_n_layers=(4, 4), vqvae_hid_dim=512,
    vqvae_heads=4, vqvae_dropout=0., vqvae_n_codebook=8, vqvae_beta=1.0,
    sparse_input_dim=54,
    num_classes=18*3*20, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(
        in_dim=vqvae_in_dim,
        n_layers=list(vqvae_n_layers),
        hid_dim=vqvae_hid_dim,
        heads=vqvae_heads,
        dropout=vqvae_dropout,
        n_codebook=vqvae_n_codebook,
        n_e=V,
        e_dim=Cvae,
        beta=vqvae_beta,
        sparse_dim=sparse_input_dim,
        patch_nums=patch_nums,
    ).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        sparse_input_dim=sparse_input_dim,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp
