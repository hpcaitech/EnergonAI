from typing import Callable
import torch
from torch import dtype
from torch import nn
from colossalai.nn import LayerNorm1D

from .mlp import MLP1D
from .attention import MultiHeadAttention1D



class Block1D(nn.Module):
    def __init__(self,
                hidden_size: int,
                num_heads: int,
                mlp_ratio: float,
                activation: Callable = nn.functional.gelu,
                layernorm_epsilon:float = 1e-5,
                dtype: dtype = torch.float16,
                bias: bool = True,
                apply_post_layernorm: bool = False,
                max_seq_len: int = 512,
                fused_qkv:bool = True,
                is_decoder:bool = True) -> None:
        super().__init__()

        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = LayerNorm1D(hidden_size, eps=layernorm_epsilon)

        self.attn = MultiHeadAttention1D(hidden_size = hidden_size,
                                              num_heads = num_heads,
                                              bias = bias,
                                              dtype = dtype,
                                              max_seq_len = max_seq_len,
                                              fused_qkv = fused_qkv,
                                              is_decoder = is_decoder)

        self.norm2 = LayerNorm1D(hidden_size, eps=layernorm_epsilon)

        self.mlp = MLP1D(hidden_size = hidden_size,
                         mlp_ratio = mlp_ratio,
                         activation = activation,
                         dtype = dtype,
                         bias = bias)
    def forward(self, hidden_states, attention_mask=None):
        
        if not self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = residual + self.attn(hidden_states, attention_mask)

        if not self.apply_post_layernorm:
            residual = hidden_states
        
        hidden_states = self.norm2(hidden_states)
        
        if self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states