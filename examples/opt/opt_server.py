import time
import torch
import uvicorn
from fastapi import FastAPI, status
from energonai.engine import InferenceEngine

from transformers import GPT2Tokenizer
from pydantic import BaseModel
from typing import Optional
from executor import Executor


class GenerationTaskReq(BaseModel):
    max_tokens: int
    prompt: str
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None


app = FastAPI()  # 创建 api 对象


@app.get("/")  # 根路由
def root():
    return {"200"}


@app.post('/generation')
async def generate(req: GenerationTaskReq):
    handle = executor.submit(req.prompt, req.max_tokens, req.top_k, req.top_p, req.temperature)
    output = await executor.wait(handle)

    return {'text': output}


@app.get("/shutdown")
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
                  log_level="info"
                  ):

    # only for the generation task
    global tokenizer
    if(tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

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
    global executor
    executor = Executor(engine, tokenizer, max_batch_size=16)
    executor.start()

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
