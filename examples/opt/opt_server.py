import time
import torch
import uvicorn
from fastapi import FastAPI, status
from energonai.engine import InferenceEngine
from energonai.server.queue_manager import QueueManager

from transformers import GPT2Tokenizer
from pydantic import BaseModel


class GenerationTaskReq(BaseModel):
    max_tokens: int
    prompt: str


app = FastAPI()  # 创建 api 对象


@app.get("/")  # 根路由
def root():
    return {"200"}


@app.post('/generation')
def generate(req: GenerationTaskReq):
    input_token = tokenizer(req.prompt, return_tensors="pt")
    total_predicted_text = req.prompt

    for i in range(1, req.max_tokens):
        output = engine.run(input_token)
        predictions = output.to_here()
        total_predicted_text += tokenizer.decode(predictions)
        # print(total_predicted_text)
        if '<|endoftext|>' in total_predicted_text:
            break
        input_token = tokenizer(total_predicted_text, return_tensors="pt")

    return {'text': total_predicted_text}

@app.post('/queue_generation', status_code=status.HTTP_200_OK)
def queue_generation(req: GenerationTaskReq):
    global batch_manager
    time_stamp = time.time()
    is_insert = batch_manager.insert_req(time_stamp, req.prompt, req.max_tokens)
    if(is_insert):
        result = batch_manager.subscribe_result(time_stamp)
    else:
        result = "Sorry, the serving is busy now." + "!" * req.max_tokens
    return {result}

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

    global batch_manager
    batch_manager = QueueManager(engine, tokenizer, max_batch_size=1, max_concurrent_user=4)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()
