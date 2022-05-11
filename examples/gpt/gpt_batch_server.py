import os
import time

import redis
import torch
import uvicorn
from transformers import GPT2Tokenizer
from fastapi import FastAPI
from fastapi import Response, Body
import torch.distributed.rpc as rpc
from energon.engine import InferenceEngine
from energon.server.batch_manager import Batch_Manager, generate_cached_cost, Manager
from energon.server.naive_batch_manager import Naive_Batch_Manager

app = FastAPI()


@app.get("/")  # 根路由
def root():
    return {"200"}

@app.post("/model_with_padding_naive")
def run_without_batch(input_str: str = Body(..., title="input_str", embed=True)):
    red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
    sub = red.pubsub()
    input_token = tokenizer(input_str, return_tensors="pt")
    time_stamp = time.time()
    naive_manager.insert_req(time_stamp, input_token, input_str)
    sub.subscribe(str(time_stamp))
    predictions = input_str
    for message in sub.listen():
        if message is not None and isinstance(message, dict):
            predictions = message.get('data')
            if not isinstance(predictions, int):
                break

    return {predictions}

@app.post("/model_with_padding")
def run(
        input_str: str = Body(..., title="input_str", embed=True)
):
    """Receive user request with post function. The input string is sent to the batch manager
    and then the result will be sent back with Redis pub-sub. The time stamp is used as the
    channel name that the current request process subscribes."""
    red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
    sub = red.pubsub()
    input_token = tokenizer(input_str, return_tensors="pt")
    time_stamp = time.time()
    batch_manager.insert_req(time_stamp, input_token, input_str)
    sub.subscribe(str(time_stamp))
    predictions = input_str
    for message in sub.listen():
        if message is not None and isinstance(message, dict):
            predictions = message.get('data')
            if not isinstance(predictions, int):
                break

    return {predictions}


@app.get("/shutdown")
async def shutdown():
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
                  rm_padding=False
                  ):
    """Initialize the tokenizer, inference engine, cached cost for current device,
    and batch manager. Then start the server."""
    if checkpoint:
        model_config = {'dtype': dtype, 'checkpoint': True, 'checkpoint_path': checkpoint}
    else:
        model_config = {'dtype': dtype}

    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

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

    global cached_cost
    cached_cost = generate_cached_cost(engine, max_seq_len=1024, max_batch_size=4, step=8, repeat_round=2, tokenizer=tokenizer)

    global batch_manager
    batch_manager = Batch_Manager(engine, cached_cost, max_batch_size=4, tokenizer=tokenizer,
                                  pad_token=GPT2Tokenizer.eos_token)

    global naive_manager
    naive_manager = Naive_Batch_Manager(engine, max_batch_size=4, tokenizer=tokenizer,
                                        pad_token=GPT2Tokenizer.eos_token)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    print("running server")
    server.run()
    print("application started")
