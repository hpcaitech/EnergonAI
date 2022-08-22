
from typing import Callable
import torch
from torch import dtype, nn
from energonai.nn import Linear1D_Col, Linear1D_Row, Classifier1D


class MLP1D(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: float,
                 activation: Callable,
                 dtype: dtype = torch.float16,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(hidden_size * mlp_ratio)
        self.dense_1 = Linear1D_Col(hidden_size, intermediate_dim, bias=bias, dtype=dtype, gather_output=False)
        self.activation = activation
        self.dense_2 = Linear1D_Row(intermediate_dim, hidden_size, bias=bias, dtype=dtype, parallel_input=True)

    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states
