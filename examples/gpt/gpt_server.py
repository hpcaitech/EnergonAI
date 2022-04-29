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

app = FastAPI()  # 创建 api 对象
# cached_cost = None
# batch_manager = Manager()
# server = None
tokenizer = GPT2Tokenizer.from_pretrained('./')


@app.get("/")  # 根路由
def root():
    return {"200"}


@app.post("/model_with_padding")
def run(
        input_str: str = Body(..., title="input_str", embed=True)
):
    global batch_manager, tokenizer
    # for the performance only
    seq_len = 512
    batch_size = 32
    red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
    sub = red.pubsub()
    input_token = tokenizer(input_str, return_tensors="pt")
    time_stamp = time.time()
    batch_manager.insert_req(time_stamp, input_token, input_str)
    sub.subscribe(str(time_stamp))
    predictions = input_str
    for message in sub.listen():
        if message is not None and isinstance(message, dict):
            print(message)
            predictions = message.get('data')
            if not isinstance(predictions, int):
                break

    # input_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.int64)
    # attention_mask = torch.randint(0, 1, (batch_size, 1, seq_len), dtype=torch.int64)
    # # seq_lens = torch.randint(1, 128, (batch_size, ), dtype=torch.int64) # generate seq_lens randomly
    # hidden_states = None
    # sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
    #
    # output = engine.run(sample)
    # output = output.to_here()
    # print(output)
    return {predictions}


@app.get("/shutdown")
async def shutdown():
    engine.clear()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


def launch_engine(model_name,
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
                  log_level="info"
                  ):
    if checkpoint:
        model_config = {'dtype': dtype, 'checkpoint': True, 'checkpoint_path': checkpoint}
    else:
        model_config = {'dtype': dtype}

    global engine
    engine = InferenceEngine(model_name,
                             model_config,
                             model_type,
                             max_batch_size=max_batch_size,
                             tp_init_size=tp_init_size,
                             pp_init_size=pp_init_size,
                             host=host,
                             port=port,
                             dtype=dtype)

    global cached_cost
    cached_cost = generate_cached_cost(engine, max_seq_len=256, max_batch_size=4, step=4, repeat_round=2)

    global batch_manager
    batch_manager = Batch_Manager(engine, cached_cost, max_seq_len=256, max_batch_size=4)
    print("batch manager initialized")

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    print("running server")
    server.run()
    print("application started")
