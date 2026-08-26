# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import math
import logging
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from einops import rearrange
from torch import nn
from VQVAE.blocks.mingpt import Block
from VQVAE.blocks.convolutions import Masked_conv, Masked_up_conv
from VQVAE.quantize import VectorQuantizer2 as VectorQuantizer
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, type='sine_frozen', max_len=1024, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        if 'sine' in type:
            rest = dim % 2
            pe = torch.zeros(max_len, dim + rest)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim + rest, 2).float() * (-math.log(10000.0) / (dim + rest)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:, :dim]
            pe = pe.unsqueeze(0)  # [1,t,d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, dim))  # [1,t,d]
            self.register_buffer('pe', pe)
        else:
            raise NameError

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class Encoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Stack(nn.Module):
    """ A stack of transformer blocks.
        Used to implement a U-net structure """

    def __init__(self, block_size, n_layer=12, n_head=8, n_embd=256,
                 dropout=0.1, causal=False, down=1, up=1,
                 pos_type='sine_frozen', sample_method='conv',
                 pos_all=False):
        super().__init__()
        config = Encoder_Config(block_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                causal=causal)
        self.drop = nn.Dropout(dropout)
        assert down == 1 or up == 1, "Unexpected combination"
        assert down in [1, 2] and up in [1, 2], "Not implemented"
        assert sample_method in ['cat', 'conv'], "Unknown sampling method"
        cat_down, slice_up = (down, up) if sample_method == 'cat' else (1, 1)
        self.cat_down, self.slice_up = cat_down, slice_up
        self.pos_all = pos_all
        self.blocks = nn.ModuleList([])
        self.pos = nn.ModuleList([])
        for i in range(config.n_layer):
            # Inside Block, standard transformer stuff happens.
            self.blocks.append(Block(config,
                                     in_factor=cat_down if i == 0 and cat_down > 1 else None,
                                     out_factor=slice_up if i == config.n_layer - 1 and slice_up > 1 else None))
            in_dim = config.n_embd * (cat_down if i == 0 and cat_down > 1 else 1)
            if pos_all or i == 0:
                self.pos.append(PositionalEncoding(dim=in_dim, max_len=block_size, type=pos_type))
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.down_conv, self.up_conv = None, None
        if sample_method == 'conv':
            if down == 2:
                self.down_conv = Masked_conv(config.n_embd, config.n_embd, pool_size=down, pool_type='max')
            elif up == 2:
                self.up_conv = Masked_up_conv(config.n_embd, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x=None):
        t = x.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.cat_down > 1:
            if self.cat_down != 2:
                raise NotImplementedError
            else:
                x = rearrange(x, 'b (t t2) c -> b t (t2 c)', t2=2)

        if self.down_conv is not None:
            x = self.down_conv(x)

        x = self.drop(x)
        for i in range(len(self.blocks)):
            x = self.pos[i](x) if (i == 0 or self.pos_all) else x
            x = self.blocks[i](x, in_residual=not (i == 0 and self.cat_down > 1),
                               out_residual=not (i == (len(self.blocks) - 1) and self.slice_up > 1))
        if self.slice_up > 1:
            x = rearrange(x, 'b t (t2 c) -> b (t t2) c', t2=2)

        if self.up_conv is not None:
            x = self.up_conv(x)

        x = self.ln_f(x)  # (bs, seq/2, 384)
        logits = self.head(x)  # (bs, seq/2, 384)
        return logits


class TransformerAutoEncoder(nn.Module):
    """
    Model composed of an encoder and a decoder.
    """

    # NOTE This is an abstract class for us
    # as we are not interested in vanilla autoencoders 
    # with low dimensionality bottlenecks, so it does not implement forward().
    def __init__(self, in_dim=1024, n_layers=[4, 4], hid_dim=256, heads=4, dropout=0.0, e_dim=1408, block_size=2048,
                 pos_type='sine_frozen', pos_all=False, sample_method='conv', sparse_dim=54):
        super().__init__()

        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        if len(hid_dim) != 1:
            raise NotImplementedError("Does not handle per-layer channel specification.")

        # Constrols masking of  attention in encoder / decoder.
        n_embd = hid_dim[0]
        self.in_dim = in_dim
        num_joints = in_dim // 6
        self.emb = nn.Linear(self.in_dim, n_embd)
        self.sparse_dim = int(sparse_dim)
        self.up_sparse = nn.Linear(self.sparse_dim, e_dim)

        # Build the encoder; basic brick is a 'Stack object'.
        self.encoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[0], n_head=heads, n_embd=n_embd,
                   dropout=dropout, down=1, pos_type=pos_type,
                   pos_all=pos_all, sample_method=sample_method)])

        # project features (hid) to latent variable dimensions (before going through bottleneck)
        # and then z to hid
        self.emb_in, self.emb_out = n_embd, e_dim
        self.quant_emb = nn.Linear(n_embd, e_dim)
        self.encoder_dim = e_dim
        self.post_quant_emb = nn.Linear(e_dim*2, n_embd)

        # Build the decoder
        self.decoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[1], n_head=heads, n_embd=n_embd,
                   up=1, pos_type=pos_type, pos_all=pos_all)])
        dim = n_embd
        # Final head to predict body and root paramaters
        self.reg_body = nn.Sequential(nn.Linear(dim, dim),
                                      nn.ReLU(), nn.Linear(dim, self.in_dim))

    def encoder(self, x):
        """ Calls each encoder stack sequentially """
        o = self.encoder_stacks[0](x)
        return o

    def decoder(self, z):
        """ Calls each decoder stack sequentially """
        o = self.decoder_stacks[0](z)
        return o

    def regressor(self, x):
        return self.reg_body(x)  # , self.reg_root(x)


class TransformerVQVAE(TransformerAutoEncoder):
    """
    Adds a quantization bottleneck to TransformerAutoEncoder.
    """

    def __init__(self, in_dim=132, n_layers=[4, 4], hid_dim=512, heads=4, dropout=0., n_codebook=8, n_e=1024,
                 e_dim=256, beta=1., sparse_dim=54, patch_nums=(5, 10, 20)):
        super().__init__(**{'in_dim': in_dim, 'n_layers': n_layers, 'hid_dim': hid_dim, 'e_dim': e_dim, 'heads': heads,
                            'dropout': dropout, 'sparse_dim': sparse_dim})
        assert e_dim % n_codebook == 0
        assert n_e % n_codebook == 0
        self.one_codebook_size = n_e // n_codebook
        self.e_dim,self.n_e=e_dim,n_e
        assert not any([a is None for a in [n_e, e_dim, beta]]), "Missing arguments"
        self.patch_nums = tuple(patch_nums)
        self.quantizer = VectorQuantizer(vocab_size=n_e, Cvae=e_dim, beta=beta, using_znorm=True, v_patch_nums=self.patch_nums)

    def forward_encoder(self, x): 
        """"
        Run the forward pass of the encoder
        """
        x_emb = self.emb(x)  # (B,T, 512)
        hid = self.encoder(x=x_emb)  # hid:(bs, seq, 512)  mask:(bs, seq/2)
        return hid

    def forward_decoder(self, z):
        bs, seq_len, *_ = z.shape
        return self.decoder_stacks[0](z)
    

    def forward(self, x,sparse):      # x-(B,T,22*6)
        batch_size, seq_len, *_ = x.size()
        hid = self.forward_encoder(x=x)  # hid:(bs, seq, 512)    mask_:(bs, seq*k)
        z = self.quant_emb(hid)# (bs, seq, 256) 
        z = rearrange(z, 'b t c -> b c t')
        z_q, usages,z_loss = self.quantize(z)  
        z_q = rearrange(z_q, 'b c t -> b t c')
        sparse_emb = self.up_sparse(sparse)  # (B,T,512)
        hid = self.post_quant_emb(torch.cat([z_q,sparse_emb],dim=-1))  # (bs,seq,512)  # this one is i.i.d
        y = self.decoder(z=hid)
        rotmat = self.regressor(y)
        rotmat = rotmat.reshape(batch_size,seq_len,22,-1)
        return rotmat, usages, z_loss

    def quantize(self, z):
        z_q, loss, indices = self.quantizer(z)  # z:(batch, seq/2, 256)
        return z_q, loss, indices

    def encode_my(self, x):
        batch_size, seq_len, *_ = x.size()
        hid = self.forward_encoder(x=x)
        return hid
    


    def decode_my(self, z_q,sparse):   # z_q: (B,T,256)
        z_q = rearrange(z_q, 'b c t -> b t c')
        batch_size, seq_len, *_ = z_q.size()
        sparse = sparse.reshape(batch_size, seq_len, -1)
        sparse_emb = self.up_sparse(sparse)  # (B,T,512)
        hid = self.post_quant_emb(torch.cat([z_q,sparse_emb],dim=-1))  # (bs,seq,512)  # this one is i.i.d
        y = self.decoder(z=hid)
        rotmat = self.regressor(y)
        rotmat = rotmat.reshape(batch_size,seq_len,22,-1)
        return rotmat
    




    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor,v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_emb(self.forward_encoder(inp_img_no_grad))
        f = rearrange(f, 'b t c -> b c t')
        return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantizer.ema_vocab_hit_SV' in state_dict and state_dict['quantizer.ema_vocab_hit_SV'].shape[0] != self.quantizer.ema_vocab_hit_SV.shape[0]:
            state_dict['quantizer.ema_vocab_hit_SV'] = self.quantizer.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    
    
if __name__=="__main__":

    model=TransformerVQVAE()
    model.cuda()
    sparse=torch.randn(1,20,54).cuda()
    x=torch.randn(1,20,132).cuda()
    out, usages,loss=model(x,sparse)
    print(out.shape,usages,loss)
