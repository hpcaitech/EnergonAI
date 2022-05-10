import os
import torch
from energon.context import Config, ParallelMode, ConfigException
from energon.core import global_context as gpc
from energon.logging import get_dist_logger


def launch(rank: int,
           world_size: int,
           host: str,
           port: int,
           backend: str = 'nccl',
           local_rank: int = None,
           seed: int = 1024,
           verbose: bool = True,
           tp_size=1,
           pp_size=1):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.
    :param rank: Rank for the default process group
    :type rank: int
    :param world_size: World size of the default process group
    :type world_size: int
    :param host: The master address for distributed training
    :type host: str
    :param port: The master port for distributed training
    :type port: str
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param local_rank: Rank for the process on the node and is used to set the default CUDA device, defaults to None.
        If local_rank = None, the default device ordinal will be calculated automatically
    :type local_rank: int, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    :raises Exception: Raise exception when config type is wrong
    """
    gpc.verbose = verbose

    # set config
    # assert isinstance(config, (Config, str, Path, dict)), \
    #     f'expected argument config to be Config, str or Path, but got {type(config)}'
    # if not isinstance(config, Config) and isinstance(config, dict):
    #     config = Config(config)
    # if isinstance(config, (str, Path)):
    #     config = Config.from_file(config)
    # config = dict(
    #     pipeline=dict(size=pp_size),
    #     tensor=dict(size=tp_size, mode='1d'),
    # )
    config = dict(parallel=dict(pipeline=dict(size=pp_size), tensor=dict(size=tp_size, mode='1d')))

    gpc.load_config(config)

    # init default process group
    gpc.init_global_dist(rank, world_size, backend, host, port)

    # init process groups for different parallel modes from config
    gpc.init_parallel_groups()

    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        gpc.set_device(local_rank)

    gpc.set_seed(seed)

    if verbose:
        logger = get_dist_logger()
        logger.info(f'Distributed environment is initialized, '
                    f'data parallel size: {gpc.data_parallel_size}, pipeline parallel size: {gpc.pipeline_parallel_size}, '
                    f'tensor parallel size: {gpc.tensor_parallel_size}', ranks=[0])


def launch_from_torch(tp_size=1,
                      pp_size=1,
                      backend: str = 'nccl',
                      seed: int = 1024,
                      verbose: bool = True):
    """A wrapper for colossalai.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch
    :type config: Union[str, dict, Config]
    :param backend: Backend for torch.distributed
    :type backend: str, optional
    :param seed: Specified random seed for every processes
    :type seed: int, optional
    :param verbose: Whether to print logs
    :type verbose: bool, optional
    """
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    launch(local_rank=local_rank,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose,
           tp_size=tp_size,
           pp_size=pp_size)


def launch_from_multiprocess(tp_size: int = 1,
                             pp_size: int = 1,
                             backend: str = 'nccl',
                             seed: int = 1024,
                             verbose: bool = True,
                             rank: int = 0,
                             local_rank: int = 0,
                             world_size: int = 1,
                             host: str = '127.0.0.1',
                             port: int = 29500
                             ):
    """A wrapper for colossalai.launch for using multiprocessing.
    As it is essential to provide a single entrance of input&&output in the triton,
    here we provide the multiprocess launch.
    TODO: only support the single node condition now. 
    """
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = f'{port}'

    launch(local_rank=local_rank,
           rank=rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend,
           seed=seed,
           verbose=verbose,
           tp_size=tp_size,
           pp_size=pp_size)
