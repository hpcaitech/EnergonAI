import logging
import torch
import uvicorn
import random
from fastapi import FastAPI, Request
from energonai.engine import InferenceEngine
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2Tokenizer
from pydantic import BaseModel, Field
from typing import Optional
from executor import Executor
from cache import ListCache, MissCacheError


class GenerationTaskReq(BaseModel):
    max_tokens: int = Field(gt=0, le=256, example=64)
    prompt: str = Field(
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.7)


app = FastAPI()


@app.post('/generation')
async def generate(data: GenerationTaskReq, request: Request):
    logger.info(f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path}" - {data}')
    key = (data.prompt, data.max_tokens)
    try:
        outputs = cache.get(key)
        output = random.choice(outputs)
        logger.info('Cache hit')
    except MissCacheError:
        inputs = tokenizer(data.prompt, truncation=True, max_length=512)
        handle = executor.submit(inputs, data.max_tokens, data.top_k, data.top_p, data.temperature)
        output = await executor.wait(handle)
        output = tokenizer.decode(output, skip_special_tokens=True)
        cache.add(key, output)
    return {'text': output}


@app.on_event("shutdown")
async def shutdown():
    executor.teardown()
    engine.clear()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


def launch_engine(model_class,
                  model_type,
                  max_batch_size: int = 1,
                  tp_init_size: int = -1,
                  pp_init_size: int = -1,
                  host: str = "localhost",
                  port: int = 29500,
                  dtype=torch.float,
                  checkpoint: str = None,
                  tokenizer_path: str = None,
                  server_host="localhost",
                  server_port=8005,
                  log_level="info",
                  allow_cors: bool = False,
                  executor_max_batch_size: int = 16,
                  cache_size: int = 50,
                  cache_list_size: int = 1,
                  fixed_cache_keys: list = [],
                  ):
    if allow_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    global logger
    logger = logging.getLogger(__name__)
    # only for the generation task
    global tokenizer
    if(tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, padding_side='left', truncation_side='left')

    if checkpoint:
        model_config = {'dtype': dtype, 'checkpoint': checkpoint}
    else:
        model_config = {'dtype': dtype}

    global engine
    engine = InferenceEngine(model_class,
                             model_config,
                             model_type,
                             max_batch_size=max_batch_size,
                             tp_init_size=tp_init_size,
                             pp_init_size=pp_init_size,
                             host=host,
                             port=port,
                             dtype=dtype)
    global cache
    cache = ListCache(cache_size, cache_list_size, fixed_keys=fixed_cache_keys)

    global executor
    executor = Executor(engine, pad_token_id=tokenizer.pad_token_id, max_batch_size=executor_max_batch_size)
    executor.start()

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
