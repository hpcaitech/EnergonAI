import os
import torch
import uvicorn
from fastapi import FastAPI
from energonai.engine import InferenceEngine
from energonai.context import MEATCONFIG

app = FastAPI() # 创建 api 对象

@app.get("/") # 根路由
def root():
    return {"200"}

@app.get("/model_with_padding")
def run():
    # for the performance only
    seq_len = 512
    batch_size = 32

    input_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.randint(0, 1, (batch_size, 1, seq_len), dtype=torch.int64)
    # seq_lens = torch.randint(1, 128, (batch_size, ), dtype=torch.int64) # generate seq_lens randomly
    hidden_states = None
    sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)

    output = engine.run(sample)
    output = output.to_here()
    print(output)
    return {"To return the string result."}

# @app.get("/model_rm_padding")
# def run():
#     # for the performance only
#     seq_len = 512
#     batch_size = 32

#     input_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.int64)
#     attention_mask = torch.randint(0, 1, (batch_size, 1, seq_len), dtype=torch.int64)
#     seq_lens = torch.randint(1, 128, (batch_size, ), dtype=torch.int) # generate seq_lens randomly
#     hidden_states = None
#     sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask, seq_lens=seq_lens)

#     output = engine.run(sample)
#     output = output.to_here()
#     print(output)
#     return {"To return the string result."}
    

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
                dtype = torch.float,
                checkpoint: str = None,
                tokenizer_path: str = None,
                server_host = "localhost",
                server_port = 8005,
                log_level = "info"
                ):
    
    if checkpoint:
        model_config = {'dtype': dtype, 'checkpoint': True, 'checkpoint_path': checkpoint}
    else:
        model_config = {'dtype': dtype}

    global engine
    engine = InferenceEngine(model_class, 
                            model_config,
                            model_type,
                            max_batch_size = max_batch_size, 
                            tp_init_size = tp_init_size, 
                            pp_init_size = pp_init_size, 
                            auto_pp = MEATCONFIG['auto_pp'],
                            host = host,
                            port = port,
                            dtype = dtype)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()