import math
import torch
from torch import nn, dtype

from colossalai.nn.layer.utils import divide
from colossalai.nn import Linear1D_Col, Linear1D_Row

from energonai.utils import get_current_device

class MultiHeadAttention1D(nn.Module):
    def __init__(self,
                hidden_size: int,
                num_heads: int,
                bias: bool = True,
                dtype: dtype = torch.float16,
                max_seq_len: int = 512,
                fused_qkv: bool = True,
                is_decoder:bool = True
                ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_heads)
        self.fused_qkv = fused_qkv
        self.is_decoder = is_decoder

        if fused_qkv:
            self.query_key_value = Linear1D_Col(hidden_size, 3 * hidden_size, bias=bias, dtype=dtype)
        else:
            self.query_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)
            self.key_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)
            self.value_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)

        self.softmax = nn.Softmax(dim=-1)

        self.dense = Linear1D_Row(hidden_size, hidden_size, bias=True, dtype=dtype, parallel_input=True)

        if is_decoder:
            self.causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8,
                                                device=get_current_device())).view(1, 1, max_seq_len, max_seq_len).bool()
            self.causal_mask_bias = torch.tensor(-1e4, dtype=dtype, device=get_current_device())

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, 
                hidden_states, 
                attention_mask=None,
                seq_lens=None):

        if self.fused_qkv:
            qkv = self.query_key_value(hidden_states)
            all_head_size = qkv.shape[-1] // 3
            num_attention_heads = divide(all_head_size, self.attention_head_size)

            qkv = self._split_heads(qkv, num_attention_heads, 3 * self.attention_head_size)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
        else:
            q = self.query_(hidden_states)
            k = self.key_(hidden_states)
            v = self.value_(hidden_states)
            all_head_size = q.shape[-1]
            num_attention_heads = divide(all_head_size, self.attention_head_size)
            q = self._split_heads(q, num_attention_heads, self.attention_head_size)
            k = self._split_heads(k, num_attention_heads, self.attention_head_size)
            v = self._split_heads(v, num_attention_heads, self.attention_head_size)

        hidden_states = torch.matmul(q, k.transpose(-1, -2))
        hidden_states = hidden_states / math.sqrt(self.attention_head_size) 
        q_len, k_len = q.size(-2), k.size(-2)

        if self.is_decoder:
            hidden_states = torch.where(self.causal_mask[: ,: ,0:q_len , 0:k_len], hidden_states, self.causal_mask_bias)
            
        if attention_mask is not None:
            hidden_states = hidden_states + attention_mask
        hidden_states = self.softmax(hidden_states)

        hidden_states = torch.matmul(hidden_states, v)

        hidden_states = hidden_states.transpose(1, 2)

        new_context_layer_shape = hidden_states.size()[:-2] + (all_head_size,)

        hidden_states = hidden_states.reshape(new_context_layer_shape)
        hidden_states = self.dense(hidden_states)

        return hidden_states


        # causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.uint8,
        #                                     device=get_current_device())).view(1, 1, q_len, k_len).bool()
        # hidden_states = torch.where(causal_mask, hidden_states, torch.tensor(-1e4, dtype=hidden_states.dtype, device=get_current_device()))