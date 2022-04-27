import os
import torch
import uvicorn
from fastapi import FastAPI
from fastapi import Response
import torch.distributed.rpc as rpc
from energon.engine import InferenceEngine
from energon.model import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from energon.model import bert_small
from batch_manager import Batch_Manager, generate_cached_cost

MODEL_CLASSES = {
    "bert_small": bert_small,
    "gpt2_small": gpt2_small,
    "gpt2_medium": gpt2_medium,
    "gpt2_large": gpt2_large,
    "gpt2_xl": gpt2_xl,
    "gpt2_8B": gpt2_8B,
    "gpt3": gpt3
}

app = FastAPI() # 创建 api 对象
engine = None
batch_wrapper = None
server = None


@app.get("/") # 根路由
def root():
    return {"200"}

@app.get("/run")
def run():
    # a string arguement to produce sample
    input_ids = torch.randint(1, 10, (32, 40), dtype=torch.int64)
    attention_mask = torch.randint(0, 1, (32, 1, 40, 40), dtype=torch.int64)
    hidden_states = None
    sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)

    output = engine.run(sample)
    output = output.to_here()
    print(output)
    return {"To return the string result."}
    

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
                dtype = torch.float,
                checkpoint: str = None,
                server_host = "localhost",
                server_port = 8005,
                log_level = "info"
                ):
    
    model_config = {'dtype': dtype, 'checkpoint': True, 'checkpoint_path': checkpoint}
    global engine
    engine = InferenceEngine(MODEL_CLASSES[model_name], 
                            model_config,
                            model_type,
                            max_batch_size = max_batch_size, 
                            tp_init_size = tp_init_size, 
                            pp_init_size = pp_init_size, 
                            host = host,
                            port = port,
                            dtype = dtype)

    global batch_wrapper
    cached_cost = generate_cached_cost(engine)
    batch_wrapper = Batch_Manager(engine, cached_cost)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()