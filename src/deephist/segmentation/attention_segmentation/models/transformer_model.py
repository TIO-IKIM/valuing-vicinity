'''
Adjusted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
'''

import torch
from torch import nn
from einops import rearrange

from src.deephist.segmentation.attention_segmentation.models.position_encoding import PositionalEncoding

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # mask logits 
        if mask is not None:
            dots = dots.masked_fill(mask == 0, -9e15)
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # mask output
        out = out.masked_fill(mask.squeeze().unsqueeze(dim=2) == 0, -9e15)
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask):
        attns = []
        for attn, ff in self.layers:
            att_x, attention = attn(x, mask=mask)
            x = att_x + x
            x = ff(x) + x
            attns.append(attention.detach())
        return x, torch.stack(attns)

class ViT(nn.Module):
    def __init__(self, 
                 kernel_size: int, 
                 dim: int, 
                 depth: int, 
                 heads: int, 
                 mlp_dim: int, 
                 hidden_dim: int,
                 att_dropout: float = 0.,
                 emb_dropout: float = 0.,
                 sin_pos_encoding: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.sin_pos_encoding = sin_pos_encoding
        
        if self.sin_pos_encoding:
            self.pos_embedding = PositionalEncoding(d_hid=dim,
                                                    n_position=kernel_size*kernel_size)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim=dim, 
                                       depth=depth, 
                                       heads=heads,
                                       dim_head=hidden_dim // heads,
                                       mlp_dim=mlp_dim,
                                       dropout=att_dropout)

    def forward(self, x, mask=None, return_attention=False):
        
        if self.sin_pos_encoding:
            x += self.pos_embedding(x, mask)  
        x = self.dropout(x)
        x, attention = self.transformer(x, mask=mask)
        
        # select center patch only
        center_x = x[:,(self.kernel_size*self.kernel_size-1)//2,:]
        if return_attention:
            return center_x, attention
        else:
            return center_x