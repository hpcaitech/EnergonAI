import click
import torch
import inspect
import energonai.server as server
import multiprocessing as mp

from energonai.context import MEATCONFIG


@click.group()
def service():
    pass


@service.command()
@click.option("--config_file", type=str)
def init(config_file):
    
    MEATCONFIG.load_config(config_file)

    click.echo(f'*** Energon Init Configurations: ***')
    for k in MEATCONFIG:
        click.echo(f'{k}\t:\t{MEATCONFIG[k]}')
    
    # prepare context for master and worker 
    world_size = MEATCONFIG['tp_init_size'] * MEATCONFIG['pp_init_size']
    num_worker = world_size - 1

    worker_port = MEATCONFIG['server_port'] + 1
    worker_rank = 1    # start from 1

    # launch each worker
    process_list = []
    mp.set_start_method('spawn')
    for i in range(num_worker):
        p = mp.Process(target=server.launch_worker,
                        args=(config_file, worker_rank + i, worker_rank + i, MEATCONFIG['server_host'], worker_port + i))
        p.start()
        process_list.append(p)

    # launch the master
    sig_server = inspect.signature(MEATCONFIG['engine_server'])
    parameters_server = sig_server.parameters

    argv = dict()
    for name, _ in parameters_server.items():
        if name in MEATCONFIG:
            argv[name] = MEATCONFIG[name]

    MEATCONFIG['engine_server'](**argv)