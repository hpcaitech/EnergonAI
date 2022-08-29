import re
from collections import OrderedDict


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
    'self_attn_layer_norm': 'norm1',
    'final_layer_norm': 'norm2',
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
        return ori_name.replace('decoder.layer_norm', 'norm')
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
        
    del new_dict['decoder.version']
    # print("="*100)
    # print(new_dict.keys())
    # print("---------------------------")
    return new_dict  # {"model": new_dict, "epoch": 0}


def id_map(matched):
    value = matched.group('value')
    return "blocks.{}.".format(value)
