import math
from typing import Callable

import os
import torch
from torch import nn as nn, Tensor, dtype

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import divide, ACT2FN
from colossalai.nn import Linear1D_Col, Linear1D_Row, LayerNorm1D, VocabParallelEmbedding1D
from energonai.kernel import transpose_pad, transpose_depad, depad
from colossalai.utils import get_current_device, is_using_pp

__all__ = [
    'BertEmbedding1D'
    'BertMLP1D',
    'BertSelfAttention1D',
    'BertTransformerLayer1D'
]

from energonai.utils.checkpointing import load_checkpoint


class BertEmbedding1D(nn.Module):
    def __init__(self,
                 embedding_dim: int,  # hidden_size
                 vocab_size: int,
                 max_position_embeddings: int,
                 num_tokentypes: int = 0,
                 padding_idx: int = 0,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding1D(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = VocabParallelEmbedding1D(max_position_embeddings, embedding_dim, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = VocabParallelEmbedding1D(num_tokentypes, embedding_dim, dtype=dtype)
        else:
            self.tokentype_embeddings = None

        self.LayerNorm = LayerNorm1D(embedding_dim, eps=layernorm_epsilon, dtype=dtype)

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        # max_padding_size = input_ids.shape[1]

        # TODO: register_buffer in advance for position_ids to speedup

        # if position_ids is None:
        #     position_ids = torch.arange(max_padding_size, dtype=torch.long, device=get_current_device()).unsqueeze(0)

        x = self.word_embeddings(input_ids)  # + self.position_embeddings(position_ids)

        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)

        x = self.LayerNorm(x)

        # if seq_lens is not None:
        #     x = depad(x, batch_size, seq_lens)

        return x


class BertSelfAttention1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention ")
        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_heads)
        self.fuse_scale_mask_softmax = fuse_scale_mask_softmax

        self.query_key_value = Linear1D_Col(hidden_size, 3 * hidden_size, bias=bias, dtype=dtype)

        if fuse_scale_mask_softmax:
            raise NotImplementedError

        self.dense = Linear1D_Row(hidden_size, hidden_size, bias=True, dtype=dtype, parallel_input=True)
        self.LayerNorm = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None):

        attention_output = self.query_key_value(hidden_states)
        all_head_size = attention_output.shape[-1] // 3
        num_attention_heads = divide(all_head_size, self.attention_head_size)  # num_heads

        new_qkv_shape = attention_output.shape[:-1] + (num_attention_heads, 3 * self.attention_head_size)
        attention_output = attention_output.view(new_qkv_shape)

        # if seq_lens is not None:
        #     # TODO: use FasterTransformer's implementation.
        #     attention_output = transpose_pad(attention_output, batch_size, max_padding_size, seq_lens,
        #                                      num_attention_heads, self.attention_head_size * 3)
        # else:
        attention_output = attention_output.permute(0, 2, 1, 3)
        # TODO: make sure self.attention_head_size*3 is correct

        q, k, v = torch.chunk(attention_output, 3, dim=-1)

        attention_output = torch.matmul(q, k.transpose(-1, -2))
        if self.fuse_scale_mask_softmax:
            raise NotImplementedError
        else:
            attention_output = attention_output / math.sqrt(self.attention_head_size)
            # if attention_mask is not None:
            #     attention_output = attention_output + attention_mask
            attention_output = nn.functional.softmax(attention_output, dim=-1)

        attention_output = torch.matmul(attention_output, v)

        # if seq_lens is not None:
        #     sum_seq = torch.sum(seq_lens)
        #     attention_output = transpose_depad(attention_output, batch_size, sum_seq, max_padding_size, seq_lens,
        #    num_attention_heads, self.attention_head_size)
        # else:
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = attention_output.size()[:-2] + (all_head_size,)
        attention_output = attention_output.reshape(new_context_layer_shape)
        attention_output = self.dense(attention_output)

        hidden_states = self.LayerNorm(attention_output + hidden_states)

        return hidden_states


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


class BertMLP1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: float,
                 activation: Callable = gelu_impl,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(hidden_size * mlp_ratio)
        self.layer_0 = Linear1D_Col(hidden_size, intermediate_dim, bias=bias, dtype=dtype, gather_output=False)
        self.activation = activation
        self.layer_1 = Linear1D_Row(intermediate_dim, hidden_size, bias=bias, dtype=dtype, parallel_input=True)
        self.LayerNorm = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)

    def forward(self, input_tensor):
        hidden_states = self.layer_0(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_1(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertTransformerLayer1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable = gelu_impl,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False):
        super().__init__()

        self.attention = BertSelfAttention1D(hidden_size,
                                             num_heads,
                                             bias,
                                             fuse_scale_mask_softmax,
                                             layernorm_epsilon,
                                             dtype)
        self.mlp = BertMLP1D(hidden_size,
                             mlp_ratio,
                             activation,
                             layernorm_epsilon,
                             dtype,
                             bias)

    def forward(self, hidden_states, attention_mask):

        batch_size = hidden_states.shape[0]

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * - 10000.0

        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.mlp(hidden_states)

        return hidden_states


class Bert1D(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = 0,
                 dtype: dtype = None,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 ):
        super().__init__()
        self.embed = BertEmbedding1D(embedding_dim=hidden_size,
                                     vocab_size=vocab_size,
                                     max_position_embeddings=max_position_embeddings,
                                     padding_idx=padding_idx,
                                     layernorm_epsilon=layernorm_epsilon,
                                     dtype=dtype)
        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(BertTransformerLayer1D(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                layernorm_epsilon=layernorm_epsilon,
                dtype=dtype,
                bias=bias,
                fuse_scale_mask_softmax=fuse_scale_mask_softmax,)
            )

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None, seq_lens=None):

        # batch_size = input_ids.shape[0]
        # max_padding_size = input_ids.shape[1]

        hidden_states = self.embed(input_ids=input_ids, position_ids=None, tokentype_ids=None)  # , seq_lens

        for block in self.blocks:
            hidden_states = block(hidden_states=hidden_states, attention_mask=attention_mask)

        hidden_states = hidden_states[:, 1, :]

        return hidden_states


def _create_bert_model(model_kwargs):
    model = Bert1D(**model_kwargs)
    return model


def bert_small(**kwargs):
    model_kwargs = dict(hidden_size=768, depth=12, num_heads=12, **kwargs)
    return _create_bert_model(model_kwargs)


def bert_large(**kwargs):
    model_kwargs = dict(hidden_size=1024, depth=24, num_heads=16, **kwargs)
    return _create_bert_model(model_kwargs)


def bert_xl(**kwargs):
    model_kwargs = dict(hidden_size=1600, depth=48, num_heads=16, **kwargs)
    return _create_bert_model(model_kwargs)


def bert_8B(**kwargs):
    model_kwargs = dict(hidden_size=3072, depth=72, num_heads=24, **kwargs)
    return _create_bert_model(model_kwargs)


def bert_175B(**kwargs):
    model_kwargs = dict(hidden_size=12288, depth=96, num_heads=96, **kwargs)
    return _create_bert_model(model_kwargs)
