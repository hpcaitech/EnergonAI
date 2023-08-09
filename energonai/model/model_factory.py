import os
import random
import time
from typing import Callable, Optional
import sys
# sys.path.append('/data/zhangxueyuan/framexi/EnergonAI/energonai/model')
# sys.path.append('../')

from .downstream import LMHead1D
from .embedding import Embedding1D,glm_Embedding1D
from .endecoder import Block1D,GLMBlock1D

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import LayerNorm1D
from colossalai.utils import get_current_device, is_using_pp
from torch import dtype, nn
from torch.nn.utils import skip_init
from torch.nn import LayerNorm

from energonai.utils.checkpointing import load_checkpoint
from energonai.utils.checkpointing_hf_gpt2 import processing_HF_GPT
from energonai.utils.checkpointing_opt import load_175b, processing_OPT

# from .downstream import LMHead1D
# from .embedding import Embedding1D
# from .endecoder import Block1D

from energonai.utils.checkpointing_glm_opt import processing_GLM
# from energonai.utils.checkpointing_glm_gpt import processing_GLM
from transformers.modeling_utils import PreTrainedModel

""" ChatGLM model configuration """
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from .mlp import gelu_impl,default_init

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)

logger = logging.get_logger('model_factory')
logger.setLevel(level=logging.WARNING)

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
    
class ChatGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ChatGLMModel`].
    It is used to instantiate an ChatGLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the ChatGLM-6B [THUDM/ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 150528):
            Vocabulary size of the ChatGLM-6B model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ChatGLMModel`] or
            [`~TFChatGLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        inner_hidden_size (`int`, *optional*, defaults to 16384):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        max_sequence_length (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        Example:
    ```python
    >>> from configuration_chatglm import ChatGLMConfig
    >>> from modeling_chatglm import ChatGLMModel

    >>> # Initializing a ChatGLM-6B THUDM/ChatGLM-6B style configuration
    >>> configuration = ChatGLMConfig()

    >>> # Initializing a model from the THUDM/ChatGLM-6B style configuration
    >>> model = ChatGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "chatglm"

    def __init__(
            self,
            vocab_size=130528,
            hidden_size=4096,
            num_layers=28,
            num_attention_heads=32,
            layernorm_epsilon=1e-5,
            use_cache=True,
            bos_token_id=130004,
            eos_token_id=130005,
            mask_token_id=130000,
            gmask_token_id=130001,
            padding_idx=3,
            max_seq_len=2048,
            inner_hidden_size=16384,
            position_encoding_2d=True,
            quantization_bit=0,
            pre_seq_len=None,
            prefix_projection=False,
            num_tokentypes: int = 0,
            mlp_ratio: float = 4.0,
            activation: Callable = gelu_impl,
            dtype: dtype = torch.float16,
            bias: bool = True,
            apply_post_layernorm: bool = True,
            first: bool = False,
            last: bool = False,
            fused_qkv: bool = True,
            checkpoint: str = None,
            model_name: str = None,
            is_decoder: bool = True,
            disable_past_cache=False,
            vocab_parallel: bool = False,
            **kwargs
    ):
        
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_seq_len
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = padding_idx
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.num_tokentypes=num_tokentypes
        self.mlp_ratio=mlp_ratio
        self.activation=activation
        self.dtype=dtype
        self.bias=bias
        self.apply_post_layernorm=apply_post_layernorm
        self.first=first
        self.last=last
        self.fused_qkv=fused_qkv
        self.checkpoint=checkpoint
        self.model_name=model_name
        self.is_decoder=is_decoder
        self.disable_past_cache=disable_past_cache
        self.vocab_parallel=vocab_parallel

        super().__init__(
            pad_token_id=padding_idx,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )



class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, use_gmasks=None):
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [torch.cat((
                torch.zeros(context_length, dtype=torch.long, device=device),
                torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
            )) for context_length in context_lengths]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[i, context_length:] = mask_positions[i]

        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, glm_PipelineModel):
            module.gradient_checkpointing = value

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class glm_PipelineModel(ChatGLMPreTrainedModel):
    def __init__(self,
                 config:ChatGLMConfig,
                 first:bool=False,
                 last:bool=False,
                 empty_init=True,
                 **model_kwargs,
                 
                 ) -> None:
        # config=ChatGLMConfig()
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        self.hidden_size = config.hidden_size
        self.first = first
        self.last = last
        self.max_seq_len = config.max_sequence_length
        self.model_name = config.model_name
        self.bos_token_id=config.bos_token_id
        self.mask_token_id=config.mask_token_id
        self.gmask_token_id=config.gmask_token_id
        self.hidden_size_per_attention_head=self.hidden_size // config.num_attention_heads
        self.position_encoding_2d=config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.num_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.gradient_checkpointing=False

        # 在embedding位置删除postition，在block中添加旋转编码
        # if first:
        self.embed = glm_Embedding1D(hidden_size=config.hidden_size,
                                        vocab_size=config.vocab_size,
                                        max_seq_len=config.max_sequence_length,
                                        num_tokentypes=config.num_tokentypes,
                                        padding_idx=config.pad_token_id,
                                        dtype=config.dtype,
                                        vocab_parallel=config.vocab_parallel)

        self.blocks = nn.ModuleList()
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if is_using_pp() else 0
        for id_ in range(config.num_layers):
            self.blocks.add_module(f'{id_ + self.pp_rank * config.num_layers}',
                                   GLMBlock1D(hidden_size=config.hidden_size,
                                           num_heads=config.num_attention_heads,
                                           layernorm_epsilon=config.layernorm_epsilon,
                                           mlp_ratio=config.mlp_ratio,
                                           inner_hidden_size=config.inner_hidden_size,
                                           hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                                           layernorm=LayerNorm,
                                           
                                           dtype=config.dtype,
                                           bias=config.bias,
                                           position_encoding_2d=self.position_encoding_2d,
                                            empty_init=empty_init,

                                           activation=config.activation,
                                           apply_post_layernorm=config.apply_post_layernorm,
                                           max_seq_len=config.max_sequence_length,
                                           fused_qkv=config.fused_qkv,
                                           is_decoder=config.is_decoder,
                                           disable_past_cache=config.disable_past_cache))
        # if last:
        # 这是glm的最后处理两层
        self.final_layernorm = LayerNorm1D(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon, dtype=config.dtype)
        # chatglm的最后一层，在ChatGLMForConditionalGeneration类中
        self.head = LMHead1D(hidden_size=config.hidden_size, vocab_size=config.vocab_size,bias=False, dtype=config.dtype, vocab_parallel=config.vocab_parallel)
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad=False
            self.prefix_tokens=torch.arange(self.pre_seq_len).long()
            self.prefix_encoder=PrefixEncoder(config)
            self.dropout=torch.nn.Dropout(0.1)
    
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

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask=None, 
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                hidden_states=None, 
                return_dict: Optional[bool] = None,
                seq_lens=None, 
                max_tokens: Optional[int] = None, 
                top_k: Optional[int] = None, 
                top_p: Optional[float] = None, 
                temperature: Optional[float] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        hidden_states = (hidden_states if hidden_states is not None else self.config.output_hidden_states)# TODO chatglm中的out_hidden_states，写为hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        if past_key_values is None:
            if self.pre_seq_len is not None:
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0], device=input_ids.device,dtype=inputs_embeds.dtype)
            else:
                past_key_values = tuple([None] * len(self.blocks))

            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )
            if position_ids is None:
                MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
                seqs = input_ids.tolist()

                mask_positions, use_gmasks = [], []
                for seq in seqs:
                    mask_token = gMASK if gMASK in seq else MASK
                    use_gmask = mask_token == gMASK
                    mask_positions.append(seq.index(mask_token))
                    use_gmasks.append(use_gmask)

                position_ids = self.get_position_ids( input_ids, mask_positions=mask_positions,device=input_ids.device,use_gmasks=use_gmasks)

        if self.pre_seq_len is not None and attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, 1, input_ids.size(-1), self.pre_seq_len).to(attention_mask.device)
            prefix_attention_mask = (prefix_attention_mask < 0.5).bool()
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=3)

        source_hidden_states=inputs_embeds.transpose(0,1)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if hidden_states else None

        if attention_mask is None:
            attention_mask = torch.zeros(1, 1, device=input_ids.device).bool()
        else:
            attention_mask = attention_mask.to(source_hidden_states.device)

        # TODO 以下的forward部分，才是难点，前提是弄懂chatglm的流程，弄懂energon的流程
        # 大流程是相似的，但是细节有很多区别
        for i, block in enumerate(self.blocks):

            if hidden_states:
                all_hidden_states = all_hidden_states + (source_hidden_states,)
            layer_past = past_key_values[i]

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    block,
                    source_hidden_states,
                    position_ids,
                    attention_mask,
                    torch.tensor(i),
                    layer_past,
                    use_cache,
                    output_attentions
                )
            else:
                layer_ret = block(
                    source_hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    layer_id=torch.tensor(i),
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            source_hidden_states = layer_ret[0]

            if use_cache:
                presents = presents + (layer_ret[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)

        # Final layer norm.
        source_hidden_states = self.final_layernorm(source_hidden_states)

        if hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
            # return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=source_hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        # return source_hidden_states

        # batch_size = input_ids.shape[0]
        # cur_len = input_ids.shape[1]
        # tgt_len = cur_len + 1 if not max_tokens else max_tokens

        # if(cur_len >= tgt_len):
        #     return input_ids

        # first_cache = True
        # for _ in range(cur_len, tgt_len):

        #     if self.first:
        #         hidden_states = self.embed(input_ids)

        #     if attention_mask is not None:
        #         attention_unfold_mask = attention_mask.view(batch_size, -1)
        #         attention_unfold_mask = attention_unfold_mask.unsqueeze(1).unsqueeze(2)
        #         attention_unfold_mask = attention_unfold_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        #         attention_unfold_mask = (1.0 - attention_unfold_mask) * -10000.0

        #     for block in self.blocks:
        #         hidden_states = block(hidden_states=hidden_states,
        #                               attention_mask=attention_unfold_mask,
        #                               first_cache=first_cache,
        #                               position_ids)

        #     if self.last:
        #         hidden_states = self.norm(hidden_states)
        #         hidden_states = self.head(hidden_states)
        #         hidden_states = self.generate(input_ids, hidden_states, top_k=top_k,top_p=top_p, temperature=temperature)
        #     if torch.all(hidden_states == 50256):
        #         break  # hard code here for opt
        #     else:
        #         input_ids = torch.cat((input_ids, hidden_states.view(-1, 1)), 1)
        #         attention_mask = torch.cat((attention_mask, torch.ones(batch_size, 1, device=torch.cuda.current_device())), 1)

        #     first_cache = False
        # return input_ids if max_tokens else hidden_states

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

def glm_create_pipeline_model(depth: int = 48,
                          layer_partitions=None,
                          **model_kwargs):
    # pdb.set_trace()

    logger = get_dist_logger('energonai')
    pipeline_size = 0
    pipeline_rank = 0

    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        logger.info(f"pipeline_size {pipeline_size}")
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(f'pipeline_rank {pipeline_rank}')
    else:
        pipeline_size = 1
        pipeline_rank = 0

    rank = gpc.get_global_rank()

    parts = partition_uniform(depth, pipeline_size)[pipeline_rank] if layer_partitions is None else layer_partitions

    for start, end in parts:
        model_kwargs['first'] = start == 0
        model_kwargs['last'] = end == depth
        model_kwargs['depth'] = end - start

        model = glm_PipelineModel(config=ChatGLMConfig(), model_kwargs=model_kwargs).to(get_current_device())
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
    logger.info(f'Rank{rank}/{pipeline_rank} model size = {numel * 2 / 1e9} GB')

    if "checkpoint" in model_kwargs.keys() and "model_name" in model_kwargs.keys():
        start = time.time()
        assert os.path.exists(model_kwargs["checkpoint"]), "Checkpoint file not found"
        # if model_kwargs['model_name'] == 'opt-175b':
        #     load_175b(model_kwargs["checkpoint"], model)
        # else:
        preprocess_fn = None
            # if model_kwargs["model_name"] == "hf_gpt2":
            #     preprocess_fn = processing_HF_GPT
            # elif model_kwargs["model_name"] == "opt":
            #     preprocess_fn = processing_OPT
        if model_kwargs['model_name'] == 'glm':
            logger.info("选择加载了glm模型")
            preprocess_fn = processing_GLM
        logger.info("继续处理glm模型")
        logger.info(f'模型检查点{model_kwargs["checkpoint"]}')
        # logger.info(f'模型{model}')
        logger.info(f'处理函数名称{preprocess_fn}')
        load_checkpoint(model_kwargs["checkpoint"], model, preprocess_fn=preprocess_fn, **model_kwargs)
        logger.info(f'Load time: {time.time() - start:.3f} s')
    return model


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

def glm_6b(**kwargs):
    model_kwargs = dict(vocab_size=130528,
                        hidden_size=4096,
                        depth=28,
                        max_seq_len=2048,
                        num_heads=32,
                        activation=nn.functional.gelu,
                        is_decoder=True,
                        fused_qkv=False,
                        model_name="glm",
                        disable_past_cache=False,
                        vocab_parallel=True,
                        config=ChatGLMConfig,
                        **kwargs)
    return glm_create_pipeline_model(**model_kwargs)