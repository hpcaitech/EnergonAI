import os
import uvicorn
import asyncio
import argparse
from fastapi import FastAPI

import torch.distributed.rpc as rpc
from energon.initialize import launch_from_multiprocess

app = FastAPI() # 创建 api 对象


@app.get("/") # 根路由
def root():
    return {"200"}

@app.get("/start/{tp_size}")
def init(tp_size: int, pp_size: int, backend: str, seed: int, verbose: bool, rank: int, local_rank: int, host: str, port: int):
    # http://127.0.0.1:8005/start/1?pp_size=1&backend=nccl&seed=1024&verbose=true&rank=0&local_rank=0&host=localhost&port=29500
    # http://127.0.0.1:8005/start/1?pp_size=1&backend=nccl&seed=1024&verbose=true&rank=0&local_rank=0&host=localhost&port=29500
    world_size = tp_size * pp_size

    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = f'{port}'
    launch_from_multiprocess(tp_size, pp_size, backend, seed, verbose, rank, local_rank, world_size, host, port)
    WORKER_NAME = "wok{}"
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16)
    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    rpc.shutdown()
    # print(f'{WORKER_NAME.format(rank)} Start!')
    return {f'{WORKER_NAME.format(rank)} Start!'}

@app.get("/shutdown")
def shutdown():
    global server
    server.should_exit = True
    server.force_exit = True
    res = server.shutdown()


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="127.0.0.1", help="Internal host for uvicorn")
parser.add_argument("--port", type=int, default=8005, help="Internal port for uvicorn")
parser.add_argument("--log_level", type=str, default="info", help="Log Level for Uvicorn")
args = parser.parse_args()

config = uvicorn.Config(app, host = args.host, port = args.port, log_level=args.log_level)
server = uvicorn.Server(config=config)
server.run()