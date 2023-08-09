import torch
from torch import nn, dtype

from colossalai.nn.layer.utils import divide
from colossalai.nn import Linear1D_Col, Linear1D_Row
from colossalai.utils import get_current_device
import torch.nn.functional as F
import math
from .mlp import default_init
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k

def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        layer_id,
        layer_past=None,
        scaling_attention_score=True,
        use_cache=False,
):
    if layer_past is not None:#TODO 
        past_key, past_value = layer_past[0], layer_past[1]
        key_layer = torch.cat((past_key, key_layer), dim=0)
        value_layer = torch.cat((past_value, value_layer), dim=0)

    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, b, nh, hidden_size = key_layer.shape

    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    query_key_layer_scaling_coeff = float(layer_id + 1)
    if scaling_attention_score:#保持维度不变5*1*32*128
        query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]1*32*5*5
    output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]5*32*128
    key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

    matmul_result = torch.zeros(
        1, 1, 1,
        dtype=query_layer.dtype,
        device=query_layer.device,
    )

    matmul_result = torch.baddbmm(
        matmul_result,
        query_layer.transpose(0, 1),  # [b * np, sq, hn]
        key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=1.0,
    )#32*5*5

    # change view to [b, np, sq, sk]1*32*5*5,以下维度不变
    attention_scores = matmul_result.view(*output_size)

    if self.scale_mask_softmax:
        self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
    else:
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores.masked_fill_(attention_mask, -10000.0)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.type(dtype)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]1*32*5*128
    output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

    # change view [sk, b * np, hn]5*32*128
    value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]1*32*5*5
    attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

    # matmul: [b * np, sq, hn]32*5*128
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]1*32*5*128
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]5*1*4096
    new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, present, attention_probs)#5*1*4096,元组2，attention_probs是32*5*5

    return outputs

class glm_MultiHeadAttention1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 hidden_size_per_attention_head,
                 empty_init,
                 bias: bool = True,
                 dtype: dtype = torch.float16,
                 max_seq_len: int = 512,
                 fused_qkv: bool = True,
                 is_decoder: bool = True,
                 disable_past_cache:bool=False,
                 position_encoding_2d:bool=True,
                 ) -> None:
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        super().__init__()


        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size

        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        
        self.query_key_value = Linear1D_Col(hidden_size, 3 * hidden_size, bias=bias, dtype=dtype)

        self.position_encoding_2d = position_encoding_2d

        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=torch.half,
            learnable=False,
        )
        # self.scale_mask_softmax = None
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head=divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head=hidden_size_per_attention_head

        self.inner_hidden_size=num_attention_heads*self.hidden_size_per_attention_head
        # self.fused_qkv = fused_qkv

        # self.is_decoder = is_decoder
        # self.disable_past_cache = disable_past_cache
        # self.scaling = self.hidden_size_per_attention_head**-0.5


        # if fused_qkv:
        
        # else:
        #     self.query_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)
        #     self.key_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)
        #     self.value_ = Linear1D_Col(hidden_size, hidden_size, bias=bias, dtype=dtype)
        # chatglm no softmax
        # self.softmax = nn.Softmax(dim=-1)

        self.dense = Linear1D_Row(hidden_size, hidden_size, bias=bias, dtype=dtype, parallel_input=True)
        # chatglm no under context
        # if is_decoder:
        #     self.causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8,device=get_current_device())).view(1, 1, max_seq_len, max_seq_len).bool()
        #     self.causal_mask_bias = torch.tensor(-1e4, dtype=dtype, device=get_current_device())

        # self.past_cache = {}

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def last_word(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]
        return hidden_states[:, -1, :].view(batch_size, 1, hidden_size)

    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores
    
    
    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        # 要将张量的最后一维切成num_partitions分
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self,
                hidden_states:torch.Tensor,
                position_ids,
                attention_mask:torch.Tensor,
                layer_id,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                output_attentions: bool = False,
                first_cache=False,
                seq_lens=None):
        

        # if self.fused_qkv:
        # if self.disable_past_cache:
        #     mixed_raw_layer = self.query_key_value(hidden_states)
        # else:
        #     if first_cache:
        #         mixed_raw_layer = self.query_key_value(hidden_states)
        #         self.past_cache['query_key_value'] = mixed_raw_layer
        #     else:
        #         mixed_raw_layer = self.query_key_value(self.last_word(hidden_states))
        #         self.past_cache['query_key_value'] = torch.cat((self.past_cache['query_key_value'], mixed_raw_layer), 1)
        #         mixed_raw_layer = self.past_cache['query_key_value']
        # TODO 倾向于在一个进程中，不应该把qkv拆开，即此处的mixed形状应该为4*1*12288，而不是4*1*6144，不行用torch的linear，而不是colossalai的线性层
        mixed_raw_layer = self.query_key_value(hidden_states)

        new_tensor_shape = mixed_raw_layer.size()[:-1] + (self.num_attention_heads_per_partition,3 * self.hidden_size_per_attention_head,)

        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]，四维的后两维是32*128
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        # all_head_size = mixed_raw_layer.shape[-1] // 3
        # num_attention_heads = divide(all_head_size, self.hidden_size_per_attention_head)
        # kvq = self._split_heads(kvq, num_attention_heads, 3 * self.attention_head_size)
        # key_layer, value_layer, query_layer = [t.contiguous() for t in torch.chunk(mixed_raw_layer, 3, dim=-1)]
        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))#5*1*32*64
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            # 计算q1的旋转位置编码
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)#4*1*64
            position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), position_ids[:, 1, :].transpose(0, 1).contiguous()#输出都是5*1
            # 在q1和k1上应用旋转位置编码，对应于常规的位置信息,输出是5*1*32*64
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            # 在q2和k2上，对应于块的位置信息,输出是5*1*32*64
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))#5*1*32*128
        else:
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        # TODO 重点看两个TODO之间
        # [seq_len, batch, hidden_size]使用注意力函数计算上下层
        # context是上下文，present现在的状态，attention_prob是注意力权重
        context_layer, present, attention_probs = attention_fn(
            self=self,
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
            hidden_size_per_partition=self.hidden_size_per_partition,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache
        )# 5*1*4096,(5*1*32*128,5*1*32*128),32*5*5
        output = self.dense(context_layer)
        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs  # output, present, attention_probs
        
        # TODO 以上是chatglm的官方实现，以下是energon源码
        # query_layer = self._split_heads(query_layer, num_attention_heads, self.hidden_size_per_attention_head)
        # key_layer = self._split_heads(key_layer, num_attention_heads, self.hidden_size_per_attention_head)
        # value_layer = self._split_heads(value_layer, num_attention_heads, self.hidden_size_per_attention_head)

        # query_layer *= self.scaling
        # hidden_states = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # q_len, k_len = query_layer.size(-2), key_layer.size(-2)

        # if self.is_decoder:
        #     hidden_states = torch.where(self.causal_mask[:, :, 0:q_len, 0:k_len], hidden_states, self.causal_mask_bias)

        # if attention_mask is not None:
        #     hidden_states = hidden_states + attention_mask
        # dtype = hidden_states.dtype
        # hidden_states = torch.softmax(hidden_states, -1, dtype=torch.float).to(dtype)

        # hidden_states = torch.matmul(hidden_states, value_layer)

        # hidden_states = hidden_states.transpose(1, 2)

        # new_context_layer_shape = hidden_states.size()[:-2] + (all_head_size,)

        # hidden_states = hidden_states.reshape(new_context_layer_shape)

        # if self.disable_past_cache:
        #     hidden_states = self.dense(hidden_states)
        # else:
        #     if first_cache:
        #         hidden_states = self.dense(hidden_states)
        #         self.past_cache['dense'] = hidden_states
        #     else:
        #         hidden_states = self.dense(self.last_word(hidden_states))
        #         self.past_cache['dense'] = torch.cat((self.past_cache['dense'], hidden_states), 1)
        #         hidden_states = self.past_cache['dense']
        # return hidden_states


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):#dim是4096/(2*32)
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))# dim是64，生成一个0-64，步长为2的浮点数序列，然后规范为0-1，步长为2/dim
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):#TODO 
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            # learnable表示旋转位置编码是否可以学习，参与模型训练，如果参与模型训练，则每个批次是不同，不需要缓存旋转位置编码
            # 在learnable为true的情况下，max_seq_len_cached为none，相当于if条件每次都成立
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq) # 4*32
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)# 4*64
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]#4*1*64
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)

class MultiHeadAttention1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 bias: bool = True,
                 dtype: dtype = torch.float16,
                 max_seq_len: int = 512,
                 fused_qkv: bool = True,
                 is_decoder: bool = True,
                 disable_past_cache=False
                 ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_heads)
        self.fused_qkv = fused_qkv
        self.is_decoder = is_decoder
        self.disable_past_cache = disable_past_cache
        self.scaling = self.attention_head_size**-0.5
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

        self.past_cache = {}

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def last_word(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]
        return hidden_states[:, -1, :].view(batch_size, 1, hidden_size)

    def forward(self,
                hidden_states,
                attention_mask=None,
                first_cache=False,
                seq_lens=None):
        if self.fused_qkv:
            if self.disable_past_cache:
                kvq = self.query_key_value(hidden_states)
            else:
                if first_cache:
                    kvq = self.query_key_value(hidden_states)
                    self.past_cache['query_key_value'] = kvq
                else:
                    kvq = self.query_key_value(self.last_word(hidden_states))
                    self.past_cache['query_key_value'] = torch.cat((self.past_cache['query_key_value'], kvq), 1)
                    kvq = self.past_cache['query_key_value']
            all_head_size = kvq.shape[-1] // 3
            num_attention_heads = divide(all_head_size, self.attention_head_size)
            # kvq = self._split_heads(kvq, num_attention_heads, 3 * self.attention_head_size)
            k, v, q = [t.contiguous() for t in torch.chunk(kvq, 3, dim=-1)]
        else:
            if self.disable_past_cache:
                q = self.query_(hidden_states)
                k = self.key_(hidden_states)
                v = self.value_(hidden_states)
            else:
                if first_cache:
                    q = self.query_(hidden_states)
                    k = self.key_(hidden_states)
                    v = self.value_(hidden_states)
                    self.past_cache['q'] = q
                    self.past_cache['k'] = k
                    self.past_cache['v'] = v
                else:
                    q = self.query_(self.last_word(hidden_states))
                    k = self.key_(self.last_word(hidden_states))
                    v = self.value_(self.last_word(hidden_states))
                    self.past_cache['q'] = torch.cat((self.past_cache['q'], q), 1)
                    self.past_cache['k'] = torch.cat((self.past_cache['k'], k), 1)
                    self.past_cache['v'] = torch.cat((self.past_cache['v'], v), 1)
                    q = self.past_cache['q']
                    k = self.past_cache['k']
                    v = self.past_cache['v']
            all_head_size = q.shape[-1]
            num_attention_heads = divide(all_head_size, self.attention_head_size)
        q = self._split_heads(q, num_attention_heads, self.attention_head_size)
        k = self._split_heads(k, num_attention_heads, self.attention_head_size)
        v = self._split_heads(v, num_attention_heads, self.attention_head_size)

        q *= self.scaling
        hidden_states = torch.matmul(q, k.transpose(-1, -2))

        q_len, k_len = q.size(-2), k.size(-2)

        if self.is_decoder:
            hidden_states = torch.where(self.causal_mask[:, :, 0:q_len, 0:k_len], hidden_states, self.causal_mask_bias)

        if attention_mask is not None:
            hidden_states = hidden_states + attention_mask
        dtype = hidden_states.dtype
        hidden_states = torch.softmax(hidden_states, -1, dtype=torch.float).to(dtype)

        hidden_states = torch.matmul(hidden_states, v)

        hidden_states = hidden_states.transpose(1, 2)

        new_context_layer_shape = hidden_states.size()[:-2] + (all_head_size,)

        hidden_states = hidden_states.reshape(new_context_layer_shape)

        if self.disable_past_cache:
            hidden_states = self.dense(hidden_states)
        else:
            if first_cache:
                hidden_states = self.dense(hidden_states)
                self.past_cache['dense'] = hidden_states
            else:
                hidden_states = self.dense(self.last_word(hidden_states))
                self.past_cache['dense'] = torch.cat((self.past_cache['dense'], hidden_states), 1)
                hidden_states = self.past_cache['dense']
        return hidden_states
