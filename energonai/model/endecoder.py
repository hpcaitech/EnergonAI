import sys
# sys.path.append('/data/zhangxueyuan/framexi/EnergonAI/energonai/model')

from typing import Callable
import torch
from torch import dtype
from torch import nn
from torch.nn import LayerNorm
from colossalai.nn import LayerNorm1D
from .mlp import glm_MLP1D,MLP1D
from .attention import MultiHeadAttention1D,glm_MultiHeadAttention1D
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

class GLMBlock1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 inner_hidden_size,
                 hidden_size_per_attention_head,
                 empty_init,
                 num_layers=28,
                 layernorm=LayerNorm,
                 position_encoding_2d=True,
                 activation: Callable = nn.functional.gelu,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = torch.float16,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 max_seq_len: int = 512,
                 fused_qkv: bool = True,
                 is_decoder: bool = True,
                 disable_past_cache=False,
                 ) -> None:
        super().__init__()

        self.apply_post_layernorm = apply_post_layernorm
        # norm1
        self.input_layernorm = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype) # down
        self.position_encoding_2d = position_encoding_2d # down
        # 需要在此中实现旋转编码,对应chatglm中的self attention
        self.attention = glm_MultiHeadAttention1D(
                                hidden_size=hidden_size,
                                num_attention_heads=num_heads,
                                hidden_size_per_attention_head=hidden_size_per_attention_head,
                                position_encoding_2d=self.position_encoding_2d,
                                empty_init=empty_init,
                                bias=bias,
                                dtype=dtype,
                                max_seq_len=max_seq_len,
                                fused_qkv=fused_qkv,
                                is_decoder=is_decoder,
                                disable_past_cache=disable_past_cache
                                )
        
        # norm2
        self.post_attention_layernorm = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.num_layers=num_layers

        # h_4h,4h_h
        self.mlp = glm_MLP1D(hidden_size=hidden_size,
                        inner_hidden_size=inner_hidden_size,
                        bias=bias,
                         dtype=dtype,
                         empty_init=empty_init,
                         
                         mlp_ratio=mlp_ratio,
                         activation=activation,
                         disable_past_cache=disable_past_cache)

    def forward(self, 
                hidden_states:torch.Tensor,
                position_ids,
                layer_id,
                attention_mask:torch.Tensor,
                use_cache:bool=False,
                output_attentions:bool=False,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                first_cache=False, 
                seq_lens=None,
                ):
        
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size] TODO
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_outputs = self.attention(
            attention_input,
            position_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)#规范化

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions

        # if not self.apply_post_layernorm:
        #     residual = hidden_states
        # hidden_states = self.norm1(hidden_states)

        # if self.apply_post_layernorm:
        #     residual = hidden_states
        # hidden_states = residual + self.attn(hidden_states=hidden_states,
        #                                      attention_mask=attention_mask,
        #                                      first_cache=first_cache,
        #                                      position_ids)

        # if not self.apply_post_layernorm:
        #     residual = hidden_states

        # hidden_states = self.norm2(hidden_states)

        # if self.apply_post_layernorm:
        #     residual = hidden_states
        # hidden_states = residual + self.mlp(hidden_states=hidden_states,
        #                                     first_cache=first_cache)

        # return hidden_states

class Block1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable = nn.functional.gelu,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = torch.float16,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 max_seq_len: int = 512,
                 fused_qkv: bool = True,
                 is_decoder: bool = True,
                 disable_past_cache=False) -> None:
        super().__init__()

        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.attn = MultiHeadAttention1D(hidden_size=hidden_size,
                                         num_heads=num_heads,
                                         bias=bias,
                                         dtype=dtype,
                                         max_seq_len=max_seq_len,
                                         fused_qkv=fused_qkv,
                                         is_decoder=is_decoder,
                                         disable_past_cache=disable_past_cache)

        self.norm2 = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.mlp = MLP1D(hidden_size=hidden_size,
                         mlp_ratio=mlp_ratio,
                         activation=activation,
                         dtype=dtype,
                         bias=bias,
                         disable_past_cache=disable_past_cache)

    def forward(self, hidden_states, attention_mask=None, first_cache=False, seq_lens=None):

        if not self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = residual + self.attn(hidden_states=hidden_states,
                                             attention_mask=attention_mask,
                                             first_cache=first_cache)

        if not self.apply_post_layernorm:
            residual = hidden_states

        hidden_states = self.norm2(hidden_states)

        if self.apply_post_layernorm:
            residual = hidden_states
        hidden_states = residual + self.mlp(hidden_states=hidden_states,
                                            first_cache=first_cache)

        return hidden_states
