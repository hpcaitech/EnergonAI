import math
from typing import Callable

import os
import torch
from torch import nn as nn, dtype
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import divide
from colossalai.utils import get_current_device, is_using_pp
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

__all__ = [
    'BertEmbedding1D'
    'BertMLP1D',
    'BertSelfAttention1D',
    'BertTransformerLayer1D'
]

from energonai.utils.checkpointing import load_checkpoint


# class BertEmbedding1D(nn.Module):
#     def __init__(self,
#                  embedding_dim: int,  # hidden_size
#                  vocab_size: int,
#                  max_position_embeddings: int,
#                  num_tokentypes: int = 0,
#                  padding_idx: int = 0,
#                  layernorm_epsilon: float = 1e-5,
#                  dtype: dtype = None) -> None:
#         super().__init__()
#         self.word_embeddings = VocabParallelEmbedding1D(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype,
#                                                         skip_tp=True)
#         self.position_embeddings = VocabParallelEmbedding1D(max_position_embeddings, embedding_dim, dtype=dtype,
#                                                             skip_tp=True)
#         if num_tokentypes > 0:
#             self.tokentype_embeddings = VocabParallelEmbedding1D(num_tokentypes, embedding_dim, dtype=dtype)
#         else:
#             self.tokentype_embeddings = None

#         self.LayerNorm = LayerNorm1D(embedding_dim, eps=layernorm_epsilon)

#     def forward(self, input_ids, position_ids=None, tokentype_ids=None, seq_lens=None, batch_size=None,
#                 max_padding_size=None):
#         max_padding_size = input_ids.shape[1]

#         # TODO: register_buffer in advance for position_ids to speedup

#         if position_ids is None:
#             position_ids = torch.arange(max_padding_size, dtype=torch.long, device=get_current_device()).unsqueeze(0)

#         x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)

#         if self.tokentype_embeddings is not None and tokentype_ids is not None:
#             x = x + self.tokentype_embeddings(tokentype_ids)

#         x = self.LayerNorm(x)

#         if seq_lens is not None:
#             x = depad(x, batch_size, seq_lens)

#         return x


class BertSelfAttention(nn.Module):
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

        self.query = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)

        self.dense = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, attention_mask=None):

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        all_head_size = q.shape[-1]
        num_attention_heads = divide(all_head_size, self.attention_head_size)  # num_heads

        new_qkv_shape = q.shape[:-1] + (num_attention_heads, self.attention_head_size)
        q = q.view(new_qkv_shape)
        k = k.view(new_qkv_shape)
        v = v.view(new_qkv_shape)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention_output = torch.matmul(q, k.permute(0, 1, 3, 2))
        # permute(0, 1, 3, 2)transpose(-1, -2)

        attention_output = attention_output / math.sqrt(self.attention_head_size)
        attention_output = nn.functional.softmax(attention_output, dim=-1)
        attention_output = torch.matmul(attention_output, v)

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


class BertMLP(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: float,
                 activation: Callable = gelu_impl,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(hidden_size * mlp_ratio)
        self.layer_0 = nn.Linear(hidden_size, intermediate_dim, bias=bias, dtype=dtype)
        self.activation = activation
        self.layer_1 = nn.Linear(intermediate_dim, hidden_size, bias=bias, dtype=dtype)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, input_tensor):
        hidden_states = self.layer_0(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_1(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertTransformerLayer(nn.Module):
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

        self.attention = BertSelfAttention(hidden_size,
                                           num_heads,
                                           bias,
                                           fuse_scale_mask_softmax,
                                           layernorm_epsilon,
                                           dtype)
        self.mlp = BertMLP(hidden_size,
                           mlp_ratio,
                           activation,
                           layernorm_epsilon,
                           dtype,
                           bias)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.mlp(hidden_states)

        return hidden_states


class PipelineBert(nn.Module):

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
                 first: bool = False,
                 last: bool = False):
        super().__init__()
        self.first = first
        self.last = last

        # if first:
        #     self.embed = BertEmbedding1D(embedding_dim=hidden_size,
        #                                  vocab_size=vocab_size,
        #                                  max_position_embeddings=max_position_embeddings,
        #                                  padding_idx=padding_idx,
        #                                  layernorm_epsilon=layernorm_epsilon,
        #                                  dtype=dtype)
        self.blocks = nn.ModuleList()
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if is_using_pp() else 0
        for id_ in range(depth):
            self.blocks.register_module("blk_{}".format(id_ + self.pp_rank * depth),
                                        BertTransformerLayer(
                                            hidden_size=hidden_size,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            activation=activation,
                                            layernorm_epsilon=layernorm_epsilon,
                                            dtype=dtype,
                                            bias=bias,
                                            fuse_scale_mask_softmax=fuse_scale_mask_softmax,
            )
            )

    def forward(self, hidden_states=None, attention_mask=None):

        batch_size = hidden_states.shape[0]

        # if self.first:
        #     hidden_states = self.embed(input_ids=input_ids, position_ids=None, tokentype_ids=None, seq_lens=seq_lens,
        #                                batch_size=batch_size, max_padding_size=max_padding_size)  # , seq_lens

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            hidden_states = block(hidden_states=hidden_states, attention_mask=attention_mask)

        if self.last:
            hidden_states = hidden_states[:, 1, :]

        return hidden_states


def partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    logger = get_dist_logger('energonai')
    parts = [[] for _ in range(pipeline_parallel_size)]  # 4
    partition_items = num_items // num_chunks  # 96 // 2
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    return parts


def _create_bert_pipeline_model(depth=48, num_chunks=1, layer_partitions=None, **model_kwargs):
    logger = get_dist_logger('energonai')
    pipeline_size = 0
    pipeline_rank = 0
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    rank = gpc.get_global_rank()

    parts = partition_uniform(depth, pipeline_size,
                              num_chunks)[pipeline_rank] if layer_partitions is None else layer_partitions
    models = []
    for start, end in parts:
        model_kwargs['first'] = start == 0
        model_kwargs['last'] = end == depth
        model_kwargs['depth'] = end - start
        chunk = PipelineBert(**model_kwargs).to(get_current_device())
        models.append(chunk)
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()

    if "checkpoint" in model_kwargs.keys():
        if model_kwargs["checkpoint"] is True:
            if gpc.get_global_rank() == 0:
                assert "checkpoint_path" in model_kwargs.keys(), "You have to specify a file path to use checkpoint loading"
                assert os.path.exists(model_kwargs["checkpoint_path"]), "Checkpoint file not found"
            load_checkpoint(model_kwargs["checkpoint_path"], model, **model_kwargs)

    logger.info(f'Rank{rank}/{pipeline_rank} model size in FP16 = {numel * 2 / 1e9} GB')
    return model


# def bert_small(**kwargs):
#     model_kwargs = dict(hidden_size=768, depth=12, num_heads=12, **kwargs)
#     return _create_bert_pipeline_model(**model_kwargs)


def bert_large(**kwargs):
    model_kwargs = dict(hidden_size=1024, depth=24, num_heads=16, **kwargs)
    return _create_bert_pipeline_model(**model_kwargs)


# def bert_xl(**kwargs):
#     model_kwargs = dict(hidden_size=1600, depth=48, num_heads=16, **kwargs)
#     return _create_bert_pipeline_model(**model_kwargs)


# def bert_8B(**kwargs):
#     model_kwargs = dict(hidden_size=3072, depth=72, num_heads=24, **kwargs)
#     return _create_bert_pipeline_model(**model_kwargs)


# def bert_175B(**kwargs):
#     model_kwargs = dict(hidden_size=12288, depth=96, num_heads=96, **kwargs)
#     return _create_bert_pipeline_model(**model_kwargs)
