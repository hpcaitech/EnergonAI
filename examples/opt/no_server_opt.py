import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]='1,5'
# os.chdir(sys.path[0])
sys.path.append("../../")
import argparse
import logging
import random
from typing import Optional
from energonai import QueueFullError, launch_engine
from energonai.model import glm_6b , opt_6B
from transformers import GPT2Tokenizer, AutoTokenizer , AutoModel
from batch import BatchManagerForGeneration
from cache import ListCache, MissCacheError
import asyncio

class GenerationTaskReq:
    def __init__(self, max_tokens: int, prompt: str, top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None):
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

def get_model_fn(model_name: str):
    model_map = {
        'glm-6b': glm_6b,
        'opt-6.7b': opt_6B,
    }
    return model_map[model_name]

async def generate(data: GenerationTaskReq):
    logger.info('已经进入generate函数')
    key = (data.prompt, data.max_tokens)
    try:
        if cache is None:
            raise MissCacheError()
        outputs = cache.get(key)
        output = random.choice(outputs)
        logger.info('Cache hit')
    except MissCacheError:
        logger.info('not Cache hit')
        inputs = tokenizer(data.prompt, truncation=True, max_length=512)
        logger.info(f"inputs:{inputs}")
        inputs['max_tokens'] = data.max_tokens
        inputs['top_k'] = data.top_k
        inputs['top_p'] = data.top_p
        inputs['temperature'] = data.temperature
        logger.info(f"inputs:{inputs}")
        try:
            logger.info('上传Input--'*10,'\n'*2)
            uid = id(data)
            logger.info('处理uid--'*5,f'uid{uid}','\n'*2)
            engine.submit(uid, inputs)
            logger.info('提交uid和inputs---'*10,'\n'*2)
            output = await engine.wait(uid)
            logger.info(f"已经推理出output--：{output}")
            output = tokenizer.decode(output, skip_special_tokens=True)
            logger.info(f"已经解码出output--：{output}")
            if cache is not None:
                cache.add(key, output)
        except QueueFullError as e:
            raise Exception(e.args[0])
    return {'text': output}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['glm-6b','opt-6.7b'],default='opt-6.7b')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19990)
    parser.add_argument('--rpc_port', type=int, default=19980)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument('--checkpoint', default='/data2/zxy/6.7b_coloss_opt')
    parser.add_argument('--cache_size', type=int, default=0)
    parser.add_argument('--cache_list_size', type=int, default=1)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    logger.info(args)
    model_kwargs = {}
    if args.checkpoint is not None:
        model_kwargs['checkpoint'] = args.checkpoint

    tokenizer = AutoTokenizer.from_pretrained("/data2/share/opt-6.7b", trust_remote_code=True)

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
    data = GenerationTaskReq(max_tokens=200, prompt="Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:")
    result=asyncio.run(generate(data))
    logger.warning(f'result\n{result}')
