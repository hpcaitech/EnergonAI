import time
import torch
import uvicorn
from transformers import GPT2Tokenizer
from fastapi import FastAPI
from fastapi import Response, Body
from energonai.engine import InferenceEngine
from energonai.legacy_batch_mgr.dynamic_batch_manager import Dynamic_Batch_Manager

app = FastAPI()


def forward_func(input_list: list = [], seq_len: int = 0, batch_size: int = 0):
    """
    Forward run function needed for batch manager
    """
    if len(input_list) == 0:
        input_list = [("test " * seq_len)[:-1] for _ in range(batch_size)]
    input_ = tokenizer(input_list, return_tensors="pt", padding="longest")
    output_ = engine.run(input_)
    return output_


def result_process(output_):
    """
    Decode the output of the model
    """
    result = tokenizer.decode(int(output_))
    return result


@app.get("/")  # 根路由
def root():
    return {"200"}


@app.post("/gpt")
def run_new_batch(input_str: str = Body(..., title="input_str", embed=True)):
    global batch_manager
    input_token = tokenizer(input_str, return_tensors="pt")
    time_stamp = time.time()
    batch_manager.insert_req(time_stamp, input_token, input_str)
    predictions = batch_manager.subscribe_result(time_stamp)
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
    tokenizer.pad_token = GPT2Tokenizer.eos_token

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

    global batch_manager
    batch_manager = Dynamic_Batch_Manager(forward_func=forward_func, result_process=result_process)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    print("running server")
    server.run()
    print("application started")
