import uvicorn
from fastapi import FastAPI
import torch.distributed.rpc as rpc
from energonai.initialize import launch_from_multiprocess
from colossalai.logging import get_dist_logger

logger = get_dist_logger('energonai')

app = FastAPI()    
@app.get("/")
def root():
    return {"200"}



@app.get("/shutdown")
async def shutdown():
    rpc.shutdown()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


def launch_worker(host="127.0.0.1",
                  port=29500,
                  tp_init_size=1,
                  pp_init_size=1,
                  backend="nccl",
                  seed=1024,
                  verbose=True,
                  rank=0,
                  local_rank=0,
                  server_host="127.0.0.1",
                  server_port=8005,
                  log_level="info"):

    world_size = tp_init_size * pp_init_size

    launch_from_multiprocess(tp_init_size, pp_init_size, backend, seed, verbose, rank, local_rank, world_size, host,
                             port)
    WORKER_NAME = "wok{}"
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16,
    _transports=["uv"] #TODO: potentially a bug
                                                         )
    rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)

    logger.info(f'RPC STATUS: RPC of Rank: {rank} is initialized.')

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
