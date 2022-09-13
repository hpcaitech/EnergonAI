import os
import math
import torch
import random
from torch import nn as nn, Tensor, dtype
from typing import Callable


from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.utils import divide, ACT2FN
from colossalai.nn import Linear1D_Col, Linear1D_Row, Classifier1D, LayerNorm1D, VocabParallelEmbedding1D
from torch.nn import Embedding
from colossalai.utils import get_current_device, is_using_pp
from energonai.utils.checkpointing import load_checkpoint
from energonai.utils.checkpointing_hf_gpt2 import processing_HF_GPT


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
                 padding_idx: int = 0,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding1D(
            vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        # self.word_embeddings = Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = VocabParallelEmbedding1D(
            max_position_embeddings, embedding_dim, dtype=dtype)
        # self.position_embeddings = Embedding(max_position_embeddings, embedding_dim, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = VocabParallelEmbedding1D(
                num_tokentypes, embedding_dim, dtype=dtype)
            # self.tokentype_embeddings = Embedding(num_tokentypes, embedding_dim, dtype=dtype)
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
        # print("wte: {}".format(self.word_embeddings(input_ids)))
        # print("wpe: {}".format(self.position_embeddings(position_ids)))
        # print("hidden_states: {}".format(x))
        return x


class GPTSelfAttention1D(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.fuse_scale_mask_softmax = fuse_scale_mask_softmax  # TODO
        self.attention_head_size = divide(dim, num_heads)
        # self.query_key_value = Linear1D_Col(dim, 3 * dim, bias=bias, dtype=dtype)
        self.query_ = Linear1D_Col(dim, dim, bias=bias, dtype=dtype)
        self.key_ = Linear1D_Col(dim, dim, bias=bias, dtype=dtype)
        self.value_ = Linear1D_Col(dim, dim, bias=bias, dtype=dtype)
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

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, attention_mask=None):
        # print("x: {}".format(x.shape))
        # qkv = self.query_key_value(x)

        # print(f'qkv {qkv.shape}')
        q = self.query_(x)
        k = self.key_(x)
        v = self.value_(x)
        all_head_size = q.shape[-1]
        num_attention_heads = divide(all_head_size, self.attention_head_size)  # num_heads
        # print(self.attention_head_size)
        # new_qkv_shape = qkv.shape[:-1] + \
        #                 (num_attention_heads, 3 * self.attention_head_size)
        # qkv = qkv.view(new_qkv_shape)
        # qkv = qkv.permute((0, 2, 1, 3))
        # print("{} qkv: {} {}".format(gpc.get_global_rank(), qkv.shape, qkv))
        # # q, k, v = torch.chunk(qkv, 3, dim=-1)
        # q, k, v = qkv.split(all_head_size, dim=2)
        # print("{} q: {} {}".format(gpc.get_global_rank(), q.shape, q))
        # print("{} k: {} {}".format(gpc.get_global_rank(), k.shape, k))
        # print("{} v: {} {}".format(gpc.get_global_rank(), v.shape, v))
        q = self._split_heads(q, num_attention_heads, self.attention_head_size)
        k = self._split_heads(k, num_attention_heads, self.attention_head_size)
        v = self._split_heads(v, num_attention_heads, self.attention_head_size)
        # print(f'qkv {qkv.shape}')   # 6 40 128
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
        # print("{} before dense: {} {}".format(gpc.get_global_rank(), x.shape, x))
        x = self.dense(x)
        # print("after mlp: {}".format(x))

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
        # self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.norm1 = LayerNorm1D(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = GPTSelfAttention1D(dim=dim,
                                       num_heads=num_heads,
                                       bias=bias,
                                       fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                                       dtype=dtype)

        # self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.norm2 = LayerNorm1D(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = GPTMLP1D(dim=dim, mlp_ratio=mlp_ratio, activation=activation, dtype=dtype, bias=bias)

    def forward(self, x, attention_mask=None):
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        # print("{} after norm1: {}".format(gpc.get_global_rank(), x))
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        # print("{} after attn: {}".format(gpc.get_global_rank(), x))
        x = self.norm2(x)
        # print("{} after norm2: {}".format(gpc.get_global_rank(), x))
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)
        # print("{} after mlp: {}".format(gpc.get_global_rank(), x))
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
                 padding_idx: int = 0,
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
        self.blocks = nn.ModuleList()
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        for id_ in range(depth):
            self.blocks.register_module("blk_{}".format(id_ + self.pp_rank * depth),
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
            )
            )
        # self.blocks = nn.ModuleList([
        #     GPTBlock1D(
        #         dim=dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         activation=activation,
        #         layernorm_epsilon=layernorm_epsilon,
        #         dtype=dtype,
        #         bias=bias,
        #         apply_post_layernorm=apply_post_layernorm,
        #         fuse_scale_mask_softmax=fuse_scale_mask_softmax,
        #     ) for _ in range(depth)
        # ])
        # self.norm = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.norm = LayerNorm1D(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
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
            attention_mask = attention_mask.to(dtype=x.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        x = self.head(self.norm(x))

        return x


def select_top_k(predictions, k=5):
    predicted_index = random.choice(predictions[0, -1, :].sort(descending=True)[1][:10])  # .item()
    return predicted_index


class PipelineGPT1D(nn.Module):

    def __init__(self,
                 vocab_size: int = 50257,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = 0,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 first: bool = False,
                 last: bool = False,
                 **kwargs):
        super().__init__()
        self.first = first
        self.last = last
        if first:
            self.embed = GPTEmbedding1D(embedding_dim=dim,
                                        vocab_size=vocab_size,
                                        max_position_embeddings=max_position_embeddings,
                                        padding_idx=padding_idx,
                                        dtype=dtype)
        self.blocks = nn.ModuleList()
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if is_using_pp() else 0
        for id_ in range(depth):
            self.blocks.register_module("blk_{}".format(id_ + self.pp_rank * depth),
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
            )
            )
        # self.blocks = nn.ModuleList([
        #     GPTBlock1D(
        #         dim=dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         activation=activation,
        #         layernorm_epsilon=layernorm_epsilon,
        #         dtype=dtype,
        #         bias=bias,
        #         apply_post_layernorm=apply_post_layernorm,
        #         fuse_scale_mask_softmax=fuse_scale_mask_softmax,
        #     ) for _ in range(depth)
        # ])
        if self.last:
            # self.norm = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            self.norm = LayerNorm1D(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            self.head = GPTLMHead1D(dim=dim, vocab_size=vocab_size,
                                    dtype=dtype)  # word_embeeding_weight=self.embed.word_embedding_weight not in the same process

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None, seq_lens=None):
        topk = 5  # TODO: add as a parameter
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
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        # print("processed attention mask: {}".format(attention_mask))
        clk_cnt = 0
        for block in self.blocks:
            # print("="*30)
            # print("processing blk {}".format(clk_cnt))
            clk_cnt += 1
            hidden_states, attention_mask = block(hidden_states, attention_mask)

        if self.last:
            hidden_states = self.head(self.norm(hidden_states))
            hidden_states = select_top_k(hidden_states, k=topk)

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


def _create_gpt_pipeline_model(depth=48, num_chunks=1, layer_partitions=None, **model_kwargs):
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
    if "checkpoint" in model_kwargs.keys():
        if model_kwargs["checkpoint"] is True:
            if gpc.get_global_rank() == 0:
                assert "checkpoint_path" in model_kwargs.keys(), "You have to specify a file path to use checkpoint loading"
                print(model_kwargs["checkpoint_path"])
                assert os.path.exists(model_kwargs["checkpoint_path"]), "Checkpoint file not found"
            load_checkpoint(model_kwargs["checkpoint_path"], model, preprocess_fn=processing_HF_GPT, **model_kwargs)
    logger.info(f'Rank{rank}/{pipeline_rank} model size = {numel * 2 / 1e9} GB')
    return model


def hf_gpt2(**kwargs):
    model_kwargs = dict(dim=768, depth=12, num_heads=12, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)
