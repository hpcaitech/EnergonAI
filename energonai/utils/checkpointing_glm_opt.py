import os
import re
from collections import OrderedDict
from typing import Dict
import pdb
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
# from transformers.utils import logging
import logging
from colossalai.logging import get_dist_logger

# logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_dist_logger('process_glm_weigth_name')
# logger.setLevel(logging.DEBUG)

__all__ = [
    'processing_GLM'
]

name_map = {
    # 'embed_tokens': 'embed.word_embeddings',
    # 'embed_positions': 'embed.position_embeddings',
    # 'layers': 'blocks',
    # 'self_attn.q_proj': 'attn.query_',
    # 'self_attn.k_proj': 'attn.key_',
    # 'self_attn.v_proj': 'attn.value_',

    'input_layernorm':'input_layernorm.module',
    
    # 'final_layer_norm': 'norm2.module',
    # 'attention.query_key_value':'attn.query_key_value',
    # 'attention.rotary_emb.inv_freq':'attn.rotary_emb.inv_freq',
    # 'attention.dense': 'attntion.dense',
    'post_attention_layernorm': 'post_attention_layernorm.module',
    'mlp.dense_h_to_4h': 'mlp.dense_1',
    'mlp.dense_4h_to_h': 'mlp.dense_2',
    
    # 'c_fc': 'dense_1',
    # 'mlp.c_proj': 'mlp.dense_2'
    # ''
}


def judge_t(key_):
    key_words = ['attn.query_key_value.weight', 'mlp.dense_1.weight', 'mlp.dense_2.weight', 'attn.dense.weight']
    for word_ in key_words:
        if word_ in key_:
            return True
    return False


def module_name_mapping(ori_name: str):
    # logger.info(ori_name)
    if ori_name == 'transformer.word_embeddings.weight':
        return "transformer.embed.word_embeddings.weight"
    # elif ori_name == 'decoder.embed_positions.weight':
    #     return "embed.position_embeddings.weight"
    elif "decoder.layer_norm" in ori_name:
        # logger.info("$"*100)
        logger.info(ori_name)
        return ori_name.replace('decoder.layer_norm', 'transformer.final_layernorm.module')
    # elif "transformer.final_layernorm" in ori_name:  # hugging face style
        # return ori_name.replace('transformer.final_layernorm', 'norm.module')
    # elif ".attn.bias" in ori_name:
    #     return ""
    else:
        # 把字符串 ori_name 中的 "transformer.layers.X." 部分替换成 "blocks.X."，然后再进行一次基于 name_map 的替换
        res = re.sub(r"transformer.layers\.(?P<value>\d+)?\.", id_map, ori_name)
        for k_ in name_map.keys():
            res = res.replace(k_, name_map[k_])
        return res
        
def id_map(matched):
    value = matched.group('value')
    return "transformer.blocks.{}.".format(value)


def processing_GLM(source_state_dict: OrderedDict):
    # pdb.set_trace()
    logger.info(len(source_state_dict))
    if 'model' in source_state_dict:
        source_state_dict = source_state_dict.pop('model')
    logger.info(len(source_state_dict))
    new_dict = OrderedDict()
    for k_ in source_state_dict.keys():
        new_k = module_name_mapping(k_)
        # logger.info(f"旧的key值{k_}，新的key值{new_k,}")
        if new_k == "":
            continue
        new_v = source_state_dict[k_]
        # new_dict[new_k] = new_v
        # glm模型不用转置，实验的看吧，这个模型是如何组织的
        # if judge_t(new_k):
        #     new_v = torch.transpose(new_v, 0, 1)
        if "attn.query_key_value.weight" in new_k:
            num_ = re.search(r"blocks\.\d+?\.", new_k)
            if num_:
                prefix = num_.group()
                # logger.info('看这里看这里')
                # logger.info(prefix)
            else:
                prefix = ''
            # logger.info("prefix: {}".format(prefix))
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
                # logger.info('看这里看这里')
                # logger.info(prefix)
            else:
                prefix = ''
            # logger.info("prefix: {}".format(prefix))
            q_, k_, v_ = torch.chunk(new_v, 3, 0)
            # logger.info(f'new shapeP{new_v.shape}')
            # logger.info(f'q_shape {q_.shape}')
            # logger.info(f'k_shape {k_.shape}')
            # logger.info(f'v_shape {v_.shape}')
            new_dict[prefix + "attn.query_.bias"] = q_
            new_dict[prefix + "attn.key_.bias"] = k_
            new_dict[prefix + "attn.value_.bias"] = v_
        else:
            new_dict[new_k] = new_v
    # logger.info(new_dict.keys())
    if 'head.dense.weight' not in new_dict:
        new_dict['head.dense.weight'] = new_dict['embed.word_embeddings.weight'].clone()

    if 'decoder.version' in new_dict:
        del new_dict['decoder.version']
    logger.info("="*100)
    # logger.info(new_dict.keys())
    logger.info(f'old dict len {len(source_state_dict)}')
    # for i,j in source_state_dict.items():
    #     logger.info(i,j.shape)
    logger.info(f'new dict len {len(new_dict)}')
    # for i,j in new_dict.items():
    #     logger.info(i,j.shape)
    logger.info("glm模型处理完毕---------------------------")
    return new_dict  # {"model": new_dict, "epoch": 0}