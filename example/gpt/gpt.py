import math
from typing import Callable

import torch
from torch import nn as nn, Tensor, dtype


from energon.context import ParallelMode
from energon.core import global_context as gpc
from energon.logging import get_dist_logger
from energon.nn.layer.utils import divide, ACT2FN
from energon.nn import Linear1D_Col, Linear1D_Row, Classifier1D
from energon.nn import VocabParallelEmbedding1D
from energon.utils import get_current_device

__all__ = [
    'GPTEmbedding1D'
    'GPTMLP1D',
    'GPTSelfAttention1D',
    'GPTTransformerLayer1D'
]

class GPTEmbedding1D(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 num_tokentypes: int = 0,
                 padding_idx: int = None,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding1D(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = VocabParallelEmbedding1D(max_position_embeddings, embedding_dim, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = VocabParallelEmbedding1D(num_tokentypes, embedding_dim, dtype=dtype)
        else:
            self.tokentype_embeddings = None

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        # padding condition, not for variable length
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=get_current_device()).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)

        return x


class GPTSelfAttention1D(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.fuse_scale_mask_softmax = fuse_scale_mask_softmax # TODO
        self.attention_head_size = divide(dim, num_heads)
        self.query_key_value = Linear1D_Col(dim, 3 * dim, bias=bias, dtype=dtype)

        if fuse_scale_mask_softmax:
            from colossalai.kernel import FusedScaleMaskSoftmax
            from colossalai.kernel.cuda_native.scaled_softmax import \
                AttnMaskType
            self.softmax = FusedScaleMaskSoftmax(input_in_fp16=True,
                                                 input_in_bf16=False,
                                                 attn_mask_type=AttnMaskType.causal,
                                                 scaled_masked_softmax_fusion=True,
                                                 mask_func=None,
                                                 softmax_in_fp32=True,
                                                 scale=math.sqrt(self.attention_head_size))
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.dense = Linear1D_Row(dim, dim, bias=True, dtype=dtype, parallel_input=True)

    def forward(self, x, attention_mask=None):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = divide(all_head_size, self.attention_head_size)
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))

        if self.fuse_scale_mask_softmax:
            x = self.softmax(x, attention_mask)
        else:
            x = x / math.sqrt(self.attention_head_size)
            # causal mask
            q_len, k_len = q.size(-2), k.size(-2)
            causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.uint8,
                                                device=get_current_device())).view(1, 1, q_len, k_len).bool()
            x = torch.where(causal_mask, x, torch.tensor(-1e4, dtype=x.dtype, device=get_current_device()))
            if attention_mask is not None:
                x = x + attention_mask
            x = self.softmax(x)


        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)

        return x


class GPTMLP1D(nn.Module):

    def __init__(self,
                 dim: int,
                 mlp_ratio: float,
                 activation: Callable,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(dim * mlp_ratio)
        self.dense_1 = Linear1D_Col(dim, intermediate_dim, bias=bias, dtype=dtype, gather_output=False)
        self.activation = activation
        self.dense_2 = Linear1D_Row(intermediate_dim, dim, bias=bias, dtype=dtype, parallel_input=True)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class GPTBlock1D(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False):
        super().__init__()
        
        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.attn = GPTSelfAttention1D(dim=dim,
                                     num_heads=num_heads,
                                     bias=bias,
                                     fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                                     dtype=dtype)

        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.mlp = GPTMLP1D(dim=dim, mlp_ratio=mlp_ratio, activation=activation, dtype=dtype, bias=bias)

    def forward(self, x, attention_mask=None):
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        x = self.norm2(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)

        return x, attention_mask


class GPTLMHead1D(nn.Module):

    def __init__(self,
                 dim: int,
                 vocab_size: int,
                 word_embeding_weight: nn.Parameter = None,
                 bias: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.dense = Classifier1D(dim, vocab_size, word_embeding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x



class GPT1D(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False) -> None:
        super().__init__()
        self.embed = GPTEmbedding1D(embedding_dim=dim,
                                  vocab_size=vocab_size,
                                  max_position_embeddings=max_position_embeddings,
                                  padding_idx=padding_idx,
                                  dtype=dtype)

        self.blocks = nn.ModuleList([
            GPTBlock1D(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                layernorm_epsilon=layernorm_epsilon,
                dtype=dtype,
                bias=bias,
                apply_post_layernorm=apply_post_layernorm,
                fuse_scale_mask_softmax=fuse_scale_mask_softmax,
            ) for _ in range(depth)
        ])       
        self.norm = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.head = GPTLMHead1D(dim=dim,
                              vocab_size=vocab_size,
                              word_embeding_weight=self.embed.word_embedding_weight,
                              dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        x = self.head(self.norm(x))

        return x


class PipelineGPT1D(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 first: bool = False,
                 last: bool = False):
        super().__init__()
        self.first = first
        self.last = last
        if first:
            self.embed = GPTEmbedding1D(embedding_dim=dim,
                                      vocab_size=vocab_size,
                                      max_position_embeddings=max_position_embeddings,
                                      padding_idx=padding_idx,
                                      dtype=dtype)

        self.blocks = nn.ModuleList([
            GPTBlock1D(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                layernorm_epsilon=layernorm_epsilon,
                dtype=dtype,
                bias=bias,
                apply_post_layernorm=apply_post_layernorm,
                fuse_scale_mask_softmax=fuse_scale_mask_softmax,
            ) for _ in range(depth)
        ])
        if self.last:
            self.norm = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            self.head = GPTLMHead1D(dim=dim, vocab_size=vocab_size, dtype=dtype)   # word_embeeding_weight=self.embed.word_embedding_weight not in the same process


    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.first:
            hidden_states = self.embed(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        if attention_mask is not None:
            if self.first:
                batch_size = input_ids.shape[0]
            else:
                batch_size = hidden_states.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)

        if self.last:
            hidden_states = self.head(self.norm(hidden_states))

        return hidden_states

def partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)] # 4
    partition_items = num_items // num_chunks # 96 // 2
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

def _create_gpt_pipeline_model(depth=48, num_chunks=1, layer_partitions=None, **model_kwargs):
    logger = get_dist_logger()
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
        chunk = PipelineGPT1D(**model_kwargs).to(get_current_device())
        models.append(chunk)
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
    logger.info(f'Rank{rank}/{pipeline_rank} model size = {numel * 2 / 1e9} GB')
    return model


def gpt2_small(**kwargs):
    model_kwargs = dict(dim=768, depth=12, num_heads=12, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)


def gpt2_medium(**kwargs):
    model_kwargs = dict(dim=1024, depth=24, num_heads=8, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)


def gpt2_large(**kwargs):
    model_kwargs = dict(dim=1536, depth=36, num_heads=12, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)

def gpt2_xl(**kwargs):
    model_kwargs = dict(dim=1600, depth=48, num_heads=16, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)

def gpt2_8B(**kwargs):
    model_kwargs = dict(dim=3072, depth=72, num_heads=24, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)

def gpt3(**kwargs):
    model_kwargs = dict(dim=12288, depth=96, num_heads=96, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)


