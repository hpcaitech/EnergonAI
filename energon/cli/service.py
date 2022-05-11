import click
import torch
import inspect
import energon.server as server
from multiprocessing import Process

from energon.context import Config


def launches(model_class=None,
             model_type=None,
             engine_server=None,
             max_batch_size=32,
             tp_init_size=1,
             pp_init_size=1,
             host="127.0.0.1",
             port=29500,
             half=True,
             checkpoint=None,
             tokenizer_path=None,
             server_host="127.0.0.1",
             server_port=8005,
             log_level="info",
             backend="nccl",
             rm_padding=False):
    click.echo(f'*** Energon Init Configurations: *** \n'
               f'Model Name: {model_class} \n'
               f'Model Type: {model_type} \n'
               f'Engine Server: {engine_server} \n'
               f'Max Batch Size: {max_batch_size} \n'
               f'Tensor Parallelism Size: {tp_init_size} \n'
               f'Pipeline Parallelism Size: {pp_init_size} \n'
               f'Communication Host: {host} \n'
               f'Communication Port: {port} \n'
               f'Is Half: {half} \n'
               f'Checkpoint Path: {checkpoint} \n'
               f'Tokenizer Path: {tokenizer_path}'
               f'Worker Server Host: {server_host} \n'
               f'Worker Server Port: {server_port} \n'
               f'Unvicorn Log Level: {log_level} \n'
               f'Remove padding: {rm_padding} \n'
               )

    if half:
        dtype = torch.half
    else:
        dtype = torch.float

    world_size = tp_init_size * pp_init_size
    num_worker = world_size - 1

    engine_port = server_port
    worker_port = server_port + 1
    worker_rank = 1  # start from 1

    process_list = []
    for i in range(num_worker):
        p = Process(target=server.launch_worker,
                    args=(host, port, tp_init_size, pp_init_size, "nccl", 1024, True, worker_rank + i, worker_rank + i,
                          server_host, worker_port + i, log_level))
        p.start()
        process_list.append(p)

    engine_server(model_class,
                  model_type,
                  max_batch_size,
                  tp_init_size,
                  pp_init_size,
                  host,
                  port,
                  dtype,
                  checkpoint,
                  tokenizer_path,
                  server_host,
                  engine_port,
                  log_level,
                  rm_padding)


@click.group()
def service():
    pass


@service.command()
@click.option("--config_file", type=str)
def init(config_file):
    cfg = Config.from_file(config_file)
    sig = inspect.signature(launches)
    parameters = sig.parameters

    argv = dict()
    for name, _ in parameters.items():
        if name in cfg:
            argv[name] = cfg[name]
    launches(**argv)
