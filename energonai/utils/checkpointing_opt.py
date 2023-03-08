import os
import re
from collections import OrderedDict
from typing import Dict

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

__all__ = [
    'processing_OPT'
]

name_map = {
    'embed_tokens': 'embed.word_embeddings',
    'embed_positions': 'embed.position_embeddings',
    # 'layers': 'blocks',
    'self_attn.q_proj': 'attn.query_',
    'self_attn.k_proj': 'attn.key_',
    'self_attn.v_proj': 'attn.value_',
    'self_attn.out_proj': 'attn.dense',
    'self_attn_layer_norm': 'norm1.module',
    'final_layer_norm': 'norm2.module',
    'fc1': 'mlp.dense_1',
    'fc2': 'mlp.dense_2'
}


def judge_t(key_):
    key_words = ['attn.query_key_value.weight', 'mlp.dense_1.weight', 'mlp.dense_2.weight', 'attn.dense.weight']
    for word_ in key_words:
        if word_ in key_:
            return True
    return False


def module_name_mapping(ori_name: str):
    # print(ori_name)
    if ori_name == 'decoder.embed_tokens.weight':
        return "embed.word_embeddings.weight"
    elif ori_name == 'decoder.embed_positions.weight':
        return "embed.position_embeddings.weight"
    elif "decoder.layer_norm" in ori_name:
        return ori_name.replace('decoder.layer_norm', 'norm.module')
    elif "decoder.final_layer_norm" in ori_name:  # hugging face style
        return ori_name.replace('decoder.final_layer_norm', 'norm.module')
    # elif ".attn.bias" in ori_name:
    #     return ""
    else:
        res = re.sub(r"decoder.layers\.(?P<value>\d+)?\.", id_map, ori_name)
        for k_ in name_map.keys():
            res = res.replace(k_, name_map[k_])
        return res


def processing_OPT(state_dict: OrderedDict):
    if 'model' in state_dict:
        state_dict = state_dict.pop('model')
    new_dict = OrderedDict()
    for k_ in state_dict.keys():
        new_k = module_name_mapping(k_)
        if new_k == "":
            continue
        new_v = state_dict[k_]
        new_dict[new_k] = new_v
        # if judge_t(new_k):
        #     new_v = torch.transpose(new_v, 0, 1)
        # if "attn.query_key_value.weight" in new_k:
        #     num_ = re.search(r"blocks\.\d+?\.", new_k)
        #     if num_:
        #         prefix = num_.group()
        #     else:
        #         prefix = ''
        #     # print("prefix: {}".format(prefix))
        #     q_, k_, v_ = torch.chunk(new_v, 3, 0)
        #     # new_dict[prefix + "attn.query_.weight"] = torch.transpose(q_, 0, 1)
        #     # new_dict[prefix + "attn.key_.weight"] = torch.transpose(k_, 0, 1)
        #     # new_dict[prefix + "attn.value_.weight"] = torch.transpose(v_, 0, 1)
        #     new_dict[prefix + "attn.query_.weight"] = q_
        #     new_dict[prefix + "attn.key_.weight"] = k_
        #     new_dict[prefix + "attn.value_.weight"] = v_
        # elif "attn.query_key_value.bias" in new_k:
        #     num_ = re.search(r"blocks\.\d+?\.", new_k)
        #     if num_:
        #         prefix = num_.group()
        #     else:
        #         prefix = ''
        #     # print("prefix: {}".format(prefix))
        #     q_, k_, v_ = torch.chunk(new_v, 3, 0)
        #     new_dict[prefix + "attn.query_.bias"] = q_
        #     new_dict[prefix + "attn.key_.bias"] = k_
        #     new_dict[prefix + "attn.value_.bias"] = v_
        # else:
        #     new_dict[new_k] = new_v
    # print(new_dict.keys())
    if 'head.dense.weight' not in new_dict:
        new_dict['head.dense.weight'] = new_dict['embed.word_embeddings.weight'].clone()

    if 'decoder.version' in new_dict:
        del new_dict['decoder.version']
    # print("="*100)
    # print(new_dict.keys())
    # print("---------------------------")
    return new_dict  # {"model": new_dict, "epoch": 0}


def id_map(matched):
    value = matched.group('value')
    return "blocks.{}.".format(value)


def preprocess_175b(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    key_map = {
        'decoder.embed_tokens.weight': 'embed.word_embeddings.weight',
        'decoder.embed_positions.weight': 'embed.position_embeddings.weight',
        'decoder.layer_norm': 'norm',
        'decoder.layers': 'blocks',
        'self_attn.qkv_proj': 'attn.query_key_value',
        'self_attn.out_proj': 'attn.dense',
        'self_attn_layer_norm': 'norm1',
        'final_layer_norm': 'norm2',
        'fc1': 'mlp.dense_1',
        'fc2': 'mlp.dense_2'
    }
    output_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for old, new in key_map.items():
            new_key = new_key.replace(old, new)
        output_sd[new_key] = v
    output_sd['head.dense.weight'] = output_sd['embed.word_embeddings.weight'].clone()
    return output_sd


def load_175b(checkpoint_dir: str, model: torch.nn.Module) -> None:
    tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    checkpoint_path = os.path.join(checkpoint_dir, f'reshard-model_part-{tp_rank}.pt')
    print(f'Rank{gpc.get_global_rank()} load {checkpoint_path}')
    state_dict = torch.load(checkpoint_path)
    state_dict = preprocess_175b(state_dict)
    for n, p in model.named_parameters():
        with torch.no_grad():
            p.copy_(state_dict[n])
