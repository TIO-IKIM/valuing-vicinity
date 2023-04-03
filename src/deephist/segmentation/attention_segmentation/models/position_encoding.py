import numpy as np

import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusiodal (fixed) positional encoding
    
    From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    """

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, mask=None):
        return x + self.pos_table.clone().detach()


class _2DPositionalEmbedding(nn.Module):
    """
    relative 2D (learnable) positional encoding
    
    From "Stand-Alone Self-Attention in Vision Models Prajit":   
    @article{ramachandran2019stand,
            title={Stand-alone self-attention in vision models},
            author={Ramachandran, Prajit and Parmar, Niki and Vaswani, Ashish and Bello, Irwan and Levskaya, Anselm and Shlens, Jon},
            journal={Advances in Neural Information Processing Systems},
            volume={32},
            year={2019}
            }
    """
    
    def __init__(self, d_hid, n_position):
        super(_2DPositionalEmbedding, self).__init__()
        
        self.d_hid = d_hid
        self.n_position = n_position
        
         # 2D position encoding: patches share a col and row pos embedding, that are concatenated
        self.pos_h = nn.Parameter(torch.randn(1, 1, 1, self.n_position, self.d_hid // 2 ), requires_grad=True) 
        self.pos_w = nn.Parameter(torch.randn(1, 1, self.n_position, 1, self.d_hid // 2 ), requires_grad=True)

        self._reset_parameters()
        
    def forward(self, k):
        batch_size, num_heads, seq_length, d_hid = k.size()

        assert self.d_hid == d_hid, 'dimensionality does not fit'
        
        # apply head-invariant relative position encoding
        k_h, k_w = k.split(self.d_hid // 2, dim=-1)
        # break up kernel to apply spatial-wise pos encoding addition
        k_h = k_h.view((batch_size, num_heads, self.n_position, self.n_position, self.d_hid // 2))
        k_w = k_w.view((batch_size, num_heads, self.n_position, self.n_position, self.d_hid // 2))
        # add relative position encoding
        k = torch.cat((k_h + self.pos_h, k_w + self.pos_w), dim=-1)
        
        k = k.view((batch_size, num_heads, seq_length, self.d_hid))
        
        return k

    def _reset_parameters(self):
        nn.init.normal_(self.pos_h, 0, 1)
        nn.init.normal_(self.pos_w, 0, 1)