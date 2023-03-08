import os
import random
import time
from typing import Callable, Optional

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import LayerNorm1D
from colossalai.utils import get_current_device, is_using_pp
from torch import dtype, nn

from energonai.utils.checkpointing import load_checkpoint
from energonai.utils.checkpointing_hf_gpt2 import processing_HF_GPT
from energonai.utils.checkpointing_opt import load_175b, processing_OPT

from .downstream import LMHead1D
from .embedding import Embedding1D
from .endecoder import Block1D

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper,
        TopPLogitsWarper)
except ImportError:
    from transformers.generation import (LogitsProcessorList,
                                         TemperatureLogitsWarper,
                                         TopKLogitsWarper, TopPLogitsWarper)


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def select_top_k(predictions, k=5):
    predicted_index = random.choice(predictions[0, -1, :].sort(descending=True)[1][:10])  # .item()
    return predicted_index


class PipelineModel(nn.Module):
    def __init__(self,
                 vocab_size: int = 50257,
                 num_tokentypes: int = 0,
                 max_seq_len: int = 512,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = gelu_impl,
                 padding_idx: int = 0,
                 dtype: dtype = torch.float16,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 first: bool = False,
                 last: bool = False,
                 fused_qkv: bool = True,
                 checkpoint: str = None,
                 model_name: str = None,
                 is_decoder: bool = True,
                 disable_past_cache=False,
                 vocab_parallel: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.first = first
        self.last = last
        self.max_seq_len = max_seq_len
        self.model_name = model_name

        if first:
            self.embed = Embedding1D(hidden_size=hidden_size,
                                     vocab_size=vocab_size,
                                     max_seq_len=max_seq_len,
                                     num_tokentypes=num_tokentypes,
                                     padding_idx=padding_idx,
                                     dtype=dtype,
                                     vocab_parallel=vocab_parallel)

        self.blocks = nn.ModuleList()
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if is_using_pp() else 0
        for id_ in range(depth):
            self.blocks.add_module(f'{id_ + self.pp_rank * depth}',
                                   Block1D(hidden_size=hidden_size,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           activation=activation,
                                           layernorm_epsilon=layernorm_epsilon,
                                           dtype=dtype,
                                           bias=bias,
                                           apply_post_layernorm=apply_post_layernorm,
                                           max_seq_len=max_seq_len,
                                           fused_qkv=fused_qkv,
                                           is_decoder=is_decoder,
                                           disable_past_cache=disable_past_cache))
        if last:
            self.norm = LayerNorm1D(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
            self.head = LMHead1D(hidden_size=hidden_size, vocab_size=vocab_size,
                                 bias=False, dtype=dtype, vocab_parallel=vocab_parallel)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None, seq_lens=None, max_tokens: Optional[int] = None, top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None):
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        tgt_len = cur_len + 1 if not max_tokens else max_tokens

        if(cur_len >= tgt_len):
            return input_ids

        first_cache = True
        for _ in range(cur_len, tgt_len):

            if self.first:
                hidden_states = self.embed(input_ids)

            if attention_mask is not None:
                attention_unfold_mask = attention_mask.view(batch_size, -1)
                attention_unfold_mask = attention_unfold_mask.unsqueeze(1).unsqueeze(2)
                attention_unfold_mask = attention_unfold_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
                attention_unfold_mask = (1.0 - attention_unfold_mask) * -10000.0

            for block in self.blocks:
                hidden_states = block(hidden_states=hidden_states,
                                      attention_mask=attention_unfold_mask,
                                      first_cache=first_cache)

            if self.last:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.head(hidden_states)
                hidden_states = self.generate(input_ids, hidden_states, top_k=top_k,
                                              top_p=top_p, temperature=temperature)
            if torch.all(hidden_states == 50256):
                break  # hard code here for opt
            else:
                input_ids = torch.cat((input_ids, hidden_states.view(-1, 1)), 1)
                attention_mask = torch.cat((attention_mask, torch.ones(
                    batch_size, 1, device=torch.cuda.current_device())), 1)

            first_cache = False
        return input_ids if max_tokens else hidden_states

    def get_logits_processor(self, top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None):
        processor_list = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            processor_list.append(TopKLogitsWarper(top_k))
        if top_p is not None and top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        return processor_list

    def generate(self, input_ids, logits, top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None):
        logits = logits[:, -1, :]
        logits_processor = self.get_logits_processor(top_k, top_p, temperature)
        logits = logits_processor(input_ids, logits)
        logits = torch.softmax(logits, -1, dtype=torch.float)
        logits = torch.multinomial(logits, num_samples=1).squeeze(1)
        return logits


def partition_uniform(num_items, pipeline_parallel_size):
    logger = get_dist_logger('energonai')
    assert num_items % pipeline_parallel_size == 0, \
        "Layer length should be divided by the number of pipeline size, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]

    base_idx = 0
    chunk_size = num_items // pipeline_parallel_size
    left = pipeline_parallel_size - num_items % pipeline_parallel_size
    if chunk_size == 0:
        logger.warning("Some nodes in Pipeline have no requests")

    for p in range(pipeline_parallel_size):
        st = base_idx
        base_idx += chunk_size + (p >= left)
        parts[p].append((st, base_idx))
    return parts


def create_pipeline_model(depth: int = 48,
                          layer_partitions=None,
                          **model_kwargs):
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

    parts = partition_uniform(depth, pipeline_size)[pipeline_rank] if layer_partitions is None else layer_partitions

    for start, end in parts:
        model_kwargs['first'] = start == 0
        model_kwargs['last'] = end == depth
        model_kwargs['depth'] = end - start
        model = PipelineModel(**model_kwargs).to(get_current_device())
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
    logger.info(f'Rank{rank}/{pipeline_rank} model size = {numel * 2 / 1e9} GB')

    if "checkpoint" in model_kwargs.keys() and "model_name" in model_kwargs.keys():
        start = time.time()
        assert os.path.exists(model_kwargs["checkpoint"]), "Checkpoint file not found"
        if model_kwargs['model_name'] == 'opt-175b':
            load_175b(model_kwargs["checkpoint"], model)
        else:
            preprocess_fn = None
            if model_kwargs["model_name"] == "hf_gpt2":
                preprocess_fn = processing_HF_GPT
            elif model_kwargs["model_name"] == "opt":
                preprocess_fn = processing_OPT
            load_checkpoint(model_kwargs["checkpoint"], model, preprocess_fn=preprocess_fn, **model_kwargs)
        logger.info(f'Load time: {time.time() - start:.3f} s')

    return model


def hf_gpt2(**kwargs):
    model_kwargs = dict(hidden_size=768,
                        depth=12,
                        max_seq_len=1024,
                        num_heads=12,
                        fused_qkv=False,
                        model_name="hf_gpt2",
                        is_decoder=True,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)


def gpt2_small(**kwargs):
    model_kwargs = dict(hidden_size=768, depth=12, num_heads=12, is_decoder=True, **kwargs)
    return create_pipeline_model(**model_kwargs)


def gpt2_large(**kwargs):
    model_kwargs = dict(hidden_size=1536, depth=36, num_heads=12, is_decoder=True, **kwargs)
    return create_pipeline_model(**model_kwargs)


def gpt2_8B(**kwargs):
    model_kwargs = dict(hidden_size=3072, depth=72, num_heads=24, is_decoder=True, **kwargs)
    return create_pipeline_model(**model_kwargs)


def gpt3(**kwargs):
    model_kwargs = dict(hidden_size=12288, depth=12, num_heads=96, is_decoder=True, **kwargs)
    return create_pipeline_model(**model_kwargs)


def bert_small(**kwargs):
    model_kwargs = dict(hidden_size=768, depth=12, num_heads=12, is_decoder=False, **kwargs)
    return create_pipeline_model(**model_kwargs)


def bert_large(**kwargs):
    model_kwargs = dict(hidden_size=1024, depth=24, num_heads=16, is_decoder=False, **kwargs)
    return create_pipeline_model(**model_kwargs)


def bert_8B(**kwargs):
    model_kwargs = dict(hidden_size=3072, depth=72, num_heads=24, is_decoder=False, **kwargs)
    return create_pipeline_model(**model_kwargs)


def bert_175B(**kwargs):
    model_kwargs = dict(hidden_size=12288, depth=96, num_heads=96, is_decoder=False, **kwargs)
    return create_pipeline_model(**model_kwargs)


def opt_125M(**kwargs):
    model_kwargs = dict(vocab_size=50272,
                        hidden_size=768,
                        depth=12,
                        max_seq_len=2050,
                        num_heads=12,
                        activation=nn.functional.relu,
                        is_decoder=True,
                        fused_qkv=False,
                        model_name="opt",
                        disable_past_cache=False,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)


def opt_6B(**kwargs):
    model_kwargs = dict(vocab_size=50272,
                        hidden_size=4096,
                        depth=32,
                        max_seq_len=2050,
                        num_heads=32,
                        activation=nn.functional.relu,
                        is_decoder=True,
                        fused_qkv=False,
                        model_name="opt",
                        disable_past_cache=False,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)


def opt_30B(**kwargs):
    model_kwargs = dict(vocab_size=50272,
                        hidden_size=7168,
                        depth=48,
                        max_seq_len=2050,
                        num_heads=56,
                        activation=nn.functional.relu,
                        is_decoder=True,
                        fused_qkv=False,
                        model_name="opt",
                        disable_past_cache=False,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)


def opt_66B(**kwargs):
    model_kwargs = dict(vocab_size=50272,
                        hidden_size=9216,
                        depth=64,
                        max_seq_len=2050,
                        num_heads=72,
                        activation=nn.functional.relu,
                        is_decoder=True,
                        fused_qkv=False,
                        model_name="opt",
                        disable_past_cache=False,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)


def opt_175B(**kwargs):
    model_kwargs = dict(vocab_size=50272,
                        hidden_size=12288,
                        depth=96,
                        max_seq_len=2050,
                        num_heads=96,
                        activation=nn.functional.relu,
                        is_decoder=True,
                        fused_qkv=True,
                        model_name="opt-175b",
                        disable_past_cache=False,
                        vocab_parallel=True,
                        **kwargs)
    return create_pipeline_model(**model_kwargs)
