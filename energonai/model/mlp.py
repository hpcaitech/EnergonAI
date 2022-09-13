
from typing import Callable, Optional
import torch
from torch import dtype, nn
from colossalai.nn import Linear1D_Col, Linear1D_Row


class MLP1D(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: float,
                 activation: Callable,
                 dtype: dtype = torch.float16,
                 bias: bool = True,
                 disable_past_cache=False):
        super().__init__()
        self.disable_past_cache = disable_past_cache
        intermediate_dim = int(hidden_size * mlp_ratio)
        self.dense_1 = Linear1D_Col(hidden_size, intermediate_dim, bias=bias, dtype=dtype, gather_output=False)
        self.activation = activation
        self.dense_2 = Linear1D_Row(intermediate_dim, hidden_size, bias=bias, dtype=dtype, parallel_input=True)
        self.past_cache = {}

    def last_word(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]
        return hidden_states[:, -1, :].view(batch_size, 1, hidden_size)

    def forward(self, hidden_states, first_cache: Optional[bool] = True):

        if self.disable_past_cache:
            hidden_states = self.dense_1(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.dense_2(hidden_states)
        else:
            if first_cache:
                hidden_states = self.dense_1(hidden_states)
                self.past_cache['dense_1'] = hidden_states
                hidden_states = self.activation(hidden_states)
                hidden_states = self.dense_2(hidden_states)
                self.past_cache['dense_2'] = hidden_states
            else:
                hidden_states = self.dense_1(self.last_word(hidden_states))
                self.past_cache['dense_1'] = torch.cat((self.past_cache['dense_1'], hidden_states), 1)
                hidden_states = self.activation(self.past_cache['dense_1'])
                hidden_states = self.dense_2(self.last_word(hidden_states))
                self.past_cache['dense_2'] = torch.cat((self.past_cache['dense_2'], hidden_states), 1)
                hidden_states = self.past_cache['dense_2']

        return hidden_states
