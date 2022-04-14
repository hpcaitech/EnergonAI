import click
import torch
import energon.server as server
from multiprocessing import  Process

@click.group()
def service():
    pass


@service.command()
@click.option("--model_name", default="bert_small", type=str)
@click.option("--model_type", default="bert", type=str)
@click.option("--max_batch_size", default=32, type=int)
@click.option("--tp_init_size", default=1, type=int)
@click.option("--pp_init_size", default=1, type=int)
@click.option("--host", default="127.0.0.1", type=str)
@click.option("--port", default=29400, type=int)
@click.option("--half", is_flag=True, show_default=True)
@click.option("--checkpoint", type=str)
@click.option("--server_host", default="127.0.0.1", type=str)
@click.option("--server_port", default=8005, type=int)
@click.option("--log_level", default="info", type=str)
@click.option("--backend", default="nccl", type=str)
def init(model_name,
         model_type,
         max_batch_size,
         tp_init_size,
         pp_init_size,
         host,
         port,
         half,
         checkpoint,
         server_host,
         server_port,
         log_level,
         backend):

    click.echo(f'*** Energon Init Configurations: *** \n'
               f'Model Name: {model_name} \n'
               f'Max Batch Size: {max_batch_size} \n'
               f'Tensor Parallelism Size: {tp_init_size} \n'
               f'Pipeline Parallelism Size: {pp_init_size} \n'
               f'Communication Host: {host} \n'
               f'Communication Port: {port} \n'
               f'Is Half: {half} \n'
               f'Checkpoint Path: {checkpoint} \n'
               f'Worker Server Host: {server_host} \n'
               f'Worker Server Port: {server_port} \n'
               f'Unvicorn Log Level: {log_level} \n')
    
    if half:
        dtype = torch.half
    else:
        dtype = torch.float

    world_size = tp_init_size * pp_init_size
    num_worker = world_size - 1
    
    engine_port = server_port
    worker_port = server_port + 1
    worker_rank = 1 # start from 1

    process_list = []
    for i in range(num_worker):
        p = Process(target=server.launch_worker, 
                    args=(host, port, tp_init_size, pp_init_size, "nccl", 1024, True, worker_rank+i, worker_rank+i, server_host, worker_port+i, log_level))
        p.start()
        process_list.append(p)
    
    server.launch_engine(model_name,
                        model_type,
                        max_batch_size,
                        tp_init_size,
                        pp_init_size,
                        host,
                        port,
                        dtype,
                        checkpoint,
                        server_host,
                        engine_port,
                        log_level)