import os
import torch
from colossalai.context import Config, ParallelMode, ConfigException
from colossalai.core import global_context as gpc
from colossalai import launch

def launch_from_multiprocess(tp_size: int = 1,
                             pp_size: int = 1,
                             backend: str = 'nccl',
                             seed: int = 1024,
                             verbose: bool = True,
                             rank: int = 0,
                             local_rank: int = 0,
                             world_size: int = 1,
                             host: str = '127.0.0.1',
                             port: int = 29500):
    """A wrapper for colossalai.launch for using multiprocessing.
    As it is essential to provide a single entrance of input&&output in the triton,
    here we provide the multiprocess launch.
    TODO: only support the single node condition now. 
    """
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = f'{port}'

    config = dict(parallel=dict(pipeline=dict(size=pp_size), tensor=dict(size=tp_size, mode='1d')))

    launch(config=config,
           local_rank=local_rank,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose)