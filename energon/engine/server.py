import os
import uvicorn
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
async def shutdown():
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()
    

def launch_worker(host="127.0.0.1", port=8005, log_level="info"):
    global server
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()