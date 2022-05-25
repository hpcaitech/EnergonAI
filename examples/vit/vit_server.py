import os
import torch
import uvicorn
from fastapi import FastAPI
from fastapi import Response
import torch.distributed.rpc as rpc
from energonai.engine import InferenceEngine
from proc_img import proc_img

app = FastAPI() # 创建 api 对象

@app.get("/") # 根路由
def root():
    return {"200"}

@app.get("/vit")
def run():
    # for the performance only
    img = proc_img('/home/lcdjs/ColossalAI-Inference/examples/vit/dataset/n01667114_9985.JPEG')
    img = img.half()
    img = torch.unsqueeze(img, 0)
    sample = dict(img=img)
    output = engine.run(sample)
    output = output.to_here()
    print(output.size())
    return {"To return the class."}

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
                            host = host,
                            port = port,
                            dtype = dtype)

    global server
    config = uvicorn.Config(app, host=server_host, port=server_port, log_level=log_level)
    server = uvicorn.Server(config=config)
    server.run()