import re
from collections import OrderedDict
import torch


__all__ = [
    'processing_HF_GPT'
]

name_map = {
    'ln_2': 'norm2',
    'c_attn': 'query_key_value',
    'attn.c_proj': 'attn.dense',
    'ln_1': 'norm1',
    'c_fc': 'dense_1',
    'mlp.c_proj': 'mlp.dense_2'
}


def judge_t(key_):
    key_words = ['attn.query_key_value.weight', 'mlp.dense_1.weight', 'mlp.dense_2.weight', 'attn.dense.weight']
    for word_ in key_words:
        if word_ in key_:
            return True
    return False


def processing_HF_GPT(state_dict: OrderedDict):
    if 'model' in state_dict:
        state_dict = state_dict.pop('model')
    new_dict = OrderedDict()
    for k_ in state_dict.keys():
        new_k = module_name_mapping(k_)
        if new_k == "":
            continue

        new_v = state_dict[k_]
        if judge_t(new_k):
            new_v = torch.transpose(new_v, 0, 1)
        if "attn.query_key_value.weight" in new_k:
            num_ = re.search(r"blocks\.\d+?\.", new_k)
            if num_:
                prefix = num_.group()
            else:
                prefix = ''
            # print("prefix: {}".format(prefix))
            q_, k_, v_ = torch.chunk(new_v, 3, 0)
            # new_dict[prefix + "attn.query_.weight"] = torch.transpose(q_, 0, 1)
            # new_dict[prefix + "attn.key_.weight"] = torch.transpose(k_, 0, 1)
            # new_dict[prefix + "attn.value_.weight"] = torch.transpose(v_, 0, 1)
            new_dict[prefix + "attn.query_.weight"] = q_
            new_dict[prefix + "attn.key_.weight"] = k_
            new_dict[prefix + "attn.value_.weight"] = v_
        elif "attn.query_key_value.bias" in new_k:
            num_ = re.search(r"blocks\.\d+?\.", new_k)
            if num_:
                prefix = num_.group()
            else:
                prefix = ''
            # print("prefix: {}".format(prefix))
            q_, k_, v_ = torch.chunk(new_v, 3, 0)
            new_dict[prefix + "attn.query_.bias"] = q_
            new_dict[prefix + "attn.key_.bias"] = k_
            new_dict[prefix + "attn.value_.bias"] = v_
        else:
            new_dict[new_k] = new_v
    new_dict['head.dense.weight'] = new_dict['embed.word_embeddings.weight'].clone()
    # print("="*100)
    # print(new_dict.keys())
    return {"model": new_dict, "epoch": 0}


def id_map(matched):
    value = matched.group('value')
    return "blocks.{}.".format(value)


def module_name_mapping(ori_name: str):
    if ori_name == 'wte.weight':
        return "embed.word_embeddings.weight"
    elif ori_name == 'wpe.weight':
        return "embed.position_embeddings.weight"
    elif "ln_f" in ori_name:
        return ori_name.replace('ln_f', 'norm')
    elif ".attn.bias" in ori_name:
        return ""
    else:
        res = re.sub(r"h\.(?P<value>\d+)?\.", id_map, ori_name)
        for k_ in name_map.keys():
            res = res.replace(k_, name_map[k_])
        return res
