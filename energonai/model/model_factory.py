import os
import random
import time
from typing import Callable, Optional
import sys

from .downstream import LMHead1D
from .embedding import Embedding1D,glm_Embedding1D
from .endecoder import Block1D,GLMBlock1D
import re
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

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
from torch.nn import CrossEntropyLoss
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig , ModelOutput
import copy
import warnings
from transformers.generation.logits_process import LogitsProcessor

logger = logging.get_logger('model_factory')
logger.setLevel(level=logging.WARNING)
_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm-6b",
    # See all ChatGLM-6B models at https://huggingface.co/models?filter=chatglm
]

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper,
        TopPLogitsWarper)
except ImportError:
    from transformers.generation import (LogitsProcessorList,
                                         TemperatureLogitsWarper,
                                         TopKLogitsWarper, TopPLogitsWarper)
    
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

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

    def forward(self,
                hidden_states=None,
                input_ids=None, 
                attention_mask=None, 
                seq_lens=None, 
                max_tokens: Optional[int] = None, 
                top_k: Optional[int] = None, 
                top_p: Optional[float] = None, 
                temperature: Optional[float] = None):
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
        if isinstance(module, ChatGLMModel):
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

CHATGLM_6B_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config ([`~ChatGLM6BConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CHATGLM_6B_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`ChatGLM6BTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
""" 
@add_start_docstrings(
    "The bare ChatGLM-6B Model transformer outputting raw hidden-states without any specific head on top.",
    CHATGLM_6B_START_DOCSTRING,
)

class ChatGLMModel(ChatGLMPreTrainedModel):
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
        self.final_layernorm = LayerNorm1D(
            normalized_shape=config.hidden_size, 
            eps=config.layernorm_epsilon, 
            dtype=config.dtype
        )

        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad=False
            self.prefix_tokens=torch.arange(self.pre_seq_len).long()
            self.prefix_encoder=PrefixEncoder(config)
            self.dropout=torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

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
    
    @add_start_docstrings_to_model_forward(CHATGLM_6B_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    
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
        self.max_seq_len = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = ChatGLMModel(config, empty_init=empty_init)
        
        self.head = LMHead1D(
            hidden_size=config.hidden_size, 
            vocab_size=config.vocab_size,bias=False, 
            dtype=config.dtype, 
            vocab_parallel=config.vocab_parallel
        )
        self.config = config
        self.quantized = False

        self.hidden_size = config.hidden_size
        self.first = first
        self.last = last
        # self.max_seq_len = config.max_sequence_length
        self.model_name = config.model_name
        self.bos_token_id=config.bos_token_id
        self.mask_token_id=config.mask_token_id
        self.gmask_token_id=config.gmask_token_id
        self.hidden_size_per_attention_head=self.hidden_size // config.num_attention_heads
        # self.position_encoding_2d=config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.num_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.gradient_checkpointing=False
        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    
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

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((*attention_mask.shape[:3], 1))], dim=3)
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = False
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, new_attention_mask], dim=2
                )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = torch.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                         zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
                else:
                    position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                                device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        else:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )
            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks
                )

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        
    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )
    
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
                temperature: Optional[float] = None,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = None,
                ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.head(hidden_states).permute(1, 0, 2).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


    @torch.no_grad()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            # generation_config = self.generation_config
            generation_config = GenerationConfig.from_model_config(self.config)

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        res=[]
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            # res.append(input_ids)
            # print(f'当前res的长度{len(res)}')
            # print(input_ids)
            # if len(input_ids) > 300:
                # break
        # return res
        return input_ids
    
    @torch.no_grad()
    def stream_chat(self,
                    # query: str, 
                    hidden_stats=None,
                    input_ids=None,
                    history: List[Tuple[str, str]] = None, 
                    max_tokens: int = 2048,
                    do_sample=True, 
                    top_p=0.7, 
                    temperature=0.95, 
                    logits_processor=None, 
                    **kwargs):
        
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_tokens, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = input_ids.to(self.device)
        inputs={"input_ids":inputs}
        outputs = self.stream_generate(**inputs, **gen_kwargs)
        # outputs = outputs[-1]
        response = self.process_response(outputs)
        return response

    def process_response(self,outputs):
        # 后处理
        return outputs

    def quantize(*kwargs):
        pass

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
        model_kwargs['return_dict']=True
        model_kwargs['output_attentions']=False
        model_kwargs['output_hidden_states']=False

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
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        **kwargs)
    return glm_create_pipeline_model(**model_kwargs)
