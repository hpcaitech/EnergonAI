import os
os.environ["CUDA_VISIBLE_DEVICES"]='3,7'
import argparse
import logging
import random
from typing import Optional
from energonai import QueueFullError, launch_engine
from energonai.model import glm_6b , opt_125M
from transformers import GPT2Tokenizer, AutoTokenizer , AutoModel
from examples.chatglm.test.input_batch import BatchManagerForGeneration
from cache import ListCache, MissCacheError
import asyncio
import pdb

class GenerationTaskReq:
    def __init__(self, max_tokens: int, prompt: str, do_sample:bool=False,top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None):
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

async def generate(data: GenerationTaskReq):
    key = (data.prompt, data.max_tokens)
    try:
        if cache is None:
            raise MissCacheError()
        outputs = cache.get(key)
        output = random.choice(outputs)
    except MissCacheError:
        if isinstance(data.prompt,list):
            pad_max_length = max(len(i) for i in data.prompt)    # 这里是按照汉字的个数，来补充token数，这是不对的，如果是英语，则会产生截断的效果
            inputs = tokenizer(data.prompt, truncation=True, max_length=pad_max_length , padding='max_length')
        else:
            inputs = tokenizer(data.prompt, truncation=True, max_length=512)
        inputs['max_tokens'] = data.max_tokens
        inputs['do_sample'] = data.do_sample
        inputs['top_k'] = data.top_k
        inputs['top_p'] = data.top_p
        inputs['temperature'] = data.temperature
        try:
            uid = id(data)
            engine.submit(uid, inputs)
            output = await engine.wait(uid)
            output = tokenizer.decode(output, skip_special_tokens=True)
            if cache is not None:
                cache.add(key, output)
        except QueueFullError as e:
            raise Exception(e.args[0])
    return {'text': output}

def get_model_fn(model_name: str):
    model_map = {
        'glm-6b': glm_6b,
    }
    return model_map[model_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['glm-6b','opt-125m'],default='glm-6b')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19997)
    parser.add_argument('--rpc_port', type=int, default=19987)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=4)
    parser.add_argument('--checkpoint', default='/data2/zxy/qkv_4_energon_glm')
    parser.add_argument('--cache_size', type=int, default=0)
    parser.add_argument('--cache_list_size', type=int, default=1)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    logger.info(args)
    model_kwargs = {}
    if args.checkpoint is not None:
        model_kwargs['checkpoint'] = args.checkpoint

    tokenizer = AutoTokenizer.from_pretrained("/data/share/chatglm-6b/", trust_remote_code=True)

    if args.cache_size > 0:
        cache = ListCache(args.cache_size, args.cache_list_size)
    else:
        cache = None
    engine = launch_engine(args.tp, 1, args.master_host, args.master_port, args.rpc_port, get_model_fn(args.model),
                           batch_manager=BatchManagerForGeneration(max_batch_size=args.max_batch_size, pad_token_id=tokenizer.pad_token_id),
                           pipe_size=args.pipe_size,
                           queue_size=args.queue_size,
                           **model_kwargs)
    # 直接调用模型生成文本
    # prompt="今天天气大概25度，有点小雨，吹着风，我想去户外散步，应该穿什么样的衣服裤子鞋子搭配。"
    prompt=['北京大学','北京航空航天大学']# 现在只能拿到一个输入的输出，不对，还得拿到第二个
    # prompt='北京大学'
    data = GenerationTaskReq(prompt=prompt,max_tokens=200)
    result=asyncio.run(generate(data))
    print(result)
