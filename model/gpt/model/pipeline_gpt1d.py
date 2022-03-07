import torch.nn as nn
import torch
import inspect

from .gpt1d import GPTTransformerLayer1D
# from .pipeline_gpt_wrapper import PipelineSharedModuleWrapper

from energon.nn import HiddenParallelEmbedding, HiddenParallelGPTLMHead1D, VocabParallelEmbedding, VocabParallelGPTLMHead1D
from energon.context import ParallelMode
from energon.core import global_context as gpc
from energon.logging import get_dist_logger

# from colossalai import kernel
# from colossalai import nn as col_nn
# import model_zoo.gpt.gpt as col_gpt


__all__ = [
    'GPT2_small_pipeline_1D',
    'GPT2_exlarge_pipeline_1D',
    # 'GPT2_exlarge_pipeline_hybrid',
    # 'GPT2_small_pipeline_hybrid',
    'GPT3_pipeline_1D',
    # 'GPT3_pipeline_hybrid',
]


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

class GenericPipelineGPT(nn.Module):
    def __init__(self, embedding=None, blocks=None, norm=None, head=None) -> None:
        super().__init__()
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.head = head
        assert blocks is not None
        if norm is not None or head is not None:
            assert norm is not None and head is not None

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids=input_ids)
        batch_size = hidden_states.shape[0]
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


class PipelineGPT1D(GenericPipelineGPT):
    def __init__(self,
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 vocab_size: int = 50304,
                 embed_drop_rate: float = 0.,
                 act_func: str = 'gelu',
                 mlp_ratio: int = 4.0,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False,
                 first: bool = False,
                 last: bool = False,
                 embed_split_hidden=False):
        embedding = None
        norm = None
        head = None
        embed_cls = VocabParallelEmbedding
        head_cls = VocabParallelGPTLMHead1D
        if embed_split_hidden:
            embed_cls = HiddenParallelEmbedding
            head_cls = HiddenParallelGPTLMHead1D
        if first:
            embedding = embed_cls(hidden_size, vocab_size, max_position_embeddings, embed_drop_rate, dtype=dtype)
        blocks = nn.ModuleList([
            GPTTransformerLayer1D(hidden_size, num_attention_heads, act_func=act_func, mlp_ratio=mlp_ratio, attention_dropout_prob=attn_drop_rate,
                                  hidden_dropout_prob=drop_rate, dtype=dtype, checkpoint=checkpoint, max_position_embeddings=max_position_embeddings,
                                  layer_norm_epsilon=layer_norm_epsilon, apply_post_layer_norm=apply_post_layer_norm)
            for _ in range(num_layers)
        ])
        if last:
            norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            head = head_cls(vocab_size=vocab_size,
                            embed_dim=hidden_size,
                            dtype=dtype)
        super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)


# class FusedPipelineGPT1D(GenericPipelineGPT):
#     def __init__(self,
#                  num_layers: int = 12,
#                  hidden_size: int = 768,
#                  num_attention_heads: int = 12,
#                  vocab_size: int = 50304,
#                  embed_drop_rate: float = 0.,
#                  act_func: str = 'gelu',
#                  mlp_ratio: int = 4.0,
#                  attn_drop_rate: float = 0.,
#                  drop_rate: float = 0.,
#                  dtype: torch.dtype = torch.float,
#                  checkpoint: bool = False,
#                  max_position_embeddings: int = 1024,
#                  layer_norm_epsilon: float = 1e-5,
#                  apply_post_layer_norm: bool = False,
#                  first: bool = False,
#                  last: bool = False,
#                  embed_split_hidden=False):
#         embedding = None
#         norm = None
#         head = None
#         embed_cls = VocabParallelEmbedding
#         head_cls = VocabParallelGPTLMHead1D
#         if embed_split_hidden:
#             embed_cls = HiddenParallelEmbedding
#             head_cls = HiddenParallelGPTLMHead1D
#         if first:
#             embedding = embed_cls(hidden_size, vocab_size, max_position_embeddings, embed_drop_rate, dtype=dtype)
#         blocks = nn.ModuleList([
#             FusedGPTTransformerLayer1D(hidden_size, num_attention_heads, act_func=act_func, mlp_ratio=mlp_ratio, attention_dropout_prob=attn_drop_rate,
#                                        hidden_dropout_prob=drop_rate, dtype=dtype, checkpoint=checkpoint, max_position_embeddings=max_position_embeddings,
#                                        layer_norm_epsilon=layer_norm_epsilon, apply_post_layer_norm=apply_post_layer_norm)
#             for _ in range(num_layers)
#         ])
#         if last:
#             norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
#             head = head_cls(vocab_size=vocab_size,
#                             embed_dim=hidden_size,
#                             dtype=dtype)
#         super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)

#     def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
#         if self.embedding is not None:
#             hidden_states = self.embedding(input_ids=input_ids)
#         attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
#         for block in self.blocks:
#             hidden_states, attention_mask = block(hidden_states, attention_mask)
#         if self.norm is not None:
#             hidden_states = self.head(self.norm(hidden_states))
#         return hidden_states


# class PipelineGPTHybrid(GenericPipelineGPT):
#     def __init__(self,
#                  num_layers: int = 12,
#                  hidden_size: int = 768,
#                  num_attention_heads: int = 12,
#                  vocab_size: int = 50304,
#                  embed_drop_rate: float = 0.,
#                  act_func: str = 'gelu',
#                  mlp_ratio: int = 4,
#                  attn_drop_rate: float = 0.,
#                  drop_rate: float = 0.,
#                  dtype: torch.dtype = torch.float,
#                  checkpoint: bool = False,
#                  max_position_embeddings: int = 1024,
#                  layer_norm_epsilon: float = 1e-5,
#                  apply_post_layer_norm: bool = False,
#                  first: bool = False,
#                  last: bool = False,
#                  embed_split_hidden=False):
#         embedding = None
#         norm = None
#         head = None
#         if first:
#             embedding = col_gpt.GPTEmbedding(
#                 hidden_size, vocab_size, max_position_embeddings, dropout=embed_drop_rate, dtype=dtype)
#         blocks = nn.ModuleList([
#             col_gpt.GPTBlock(hidden_size, num_attention_heads, mlp_ratio=mlp_ratio, attention_dropout=attn_drop_rate,
#                              dropout=drop_rate, dtype=dtype, checkpoint=checkpoint, activation=nn.functional.gelu)
#             for _ in range(num_layers)
#         ])
#         if last:
#             norm = col_nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
#             head = col_gpt.GPTLMHead(vocab_size=vocab_size,
#                                      dim=hidden_size,
#                                      dtype=dtype,
#                                      bias=False)
#         super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_generic_gpt_pipeline_1d(module_cls, num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()

    pipeline_size = 0
    pipeline_rank = 0
    rank = gpc.get_global_rank()

    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0
    
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank] # [[(0, 24)], [(24, 48)], [(48, 72)], [(72, 96)]]
    models = []

    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = end == num_layers

        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')

        chunk = module_cls(**_filter_kwargs(module_cls.__init__, kwargs)).to(device)
        models.append(chunk)

        # if start == 0:
        #     wrapper.register_module(chunk.embedding.word_embeddings)
        # elif end == num_layers:
        #     wrapper.register_module(chunk.head)
            
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
    logger.info(f'Rank{rank}/{pipeline_rank} model size = {numel * 2 / 1e9} GB')

    return model


def _build_gpt_pipeline_1d(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    model = PipelineGPT1D
    return _build_generic_gpt_pipeline_1d(model, num_layers, num_chunks, device, **kwargs)



# def _build_gpt_pipeline_1d(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
#     if gpc.is_initialized(ParallelMode.PIPELINE):
#     else:
#         model = PipelineGPT1D(kwargs, first=True,last=True)

def GPT2_small_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
    cfg = dict(hidden_size=768, num_attention_heads=12, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden)
    return _build_gpt_pipeline_1d(12, num_chunks, **cfg)


def GPT2_exlarge_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
    cfg = dict(hidden_size=1600, num_attention_heads=32, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden)
    return _build_gpt_pipeline_1d(48, num_chunks, **cfg)


def GPT3_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
    cfg = dict(hidden_size=12288, num_attention_heads=96,
               checkpoint=checkpoint, max_position_embeddings=2048, dtype=dtype, embed_split_hidden=embed_split_hidden)
    return _build_gpt_pipeline_1d(96, num_chunks, **cfg)


# def GPT2_exlarge_pipeline_hybrid(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
#     cfg = dict(hidden_size=1600, num_attention_heads=32, checkpoint=checkpoint,
#                dtype=dtype, embed_split_hidden=embed_split_hidden)
#     return _build_gpt_pipeline_hybrid(48, num_chunks, **cfg)

# def GPT2_small_pipeline_hybrid(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
#     cfg = dict(hidden_size=768, num_attention_heads=12, checkpoint=checkpoint,
#                dtype=dtype, embed_split_hidden=embed_split_hidden)
#     return _build_gpt_pipeline_hybrid(12, num_chunks, **cfg)


# def GPT3_pipeline_hybrid(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
#     cfg = dict(hidden_size=12288, num_attention_heads=96,
#                checkpoint=checkpoint, max_position_embeddings=2048, dtype=dtype, embed_split_hidden=embed_split_hidden)
#     return _build_gpt_pipeline_hybrid(96, num_chunks, **cfg)