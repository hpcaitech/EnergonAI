from collections import OrderedDict

import torch
import torch.distributed as dist

from colossalai.utils import is_using_pp
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from typing import Optional, Callable
from colossalai.utils.checkpointing import partition_pipeline_parallel_state_dict, broadcast_model


__all__ = [
    "load_checkpoint", "load_state_dict"
]

import os
from multiprocessing import Pool
from time import time


def load_state_dict(path: str):
    if os.path.isfile(path):
        return torch.load(path)
    assert os.path.isdir(path)
    state_dict = {}
    files = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    procs = int(os.environ.get('LOAD_N_PROC', '1'))
    procs = min(procs, len(files))
    print(f'load {len(files)} files using {procs} procs')
    if procs > 1:
        with Pool(procs) as pool:
            state_dicts = pool.map(torch.load, files)
        for sd in state_dicts:
            state_dict.update(sd)
    else:
        for filepath in files:
            sd = torch.load(filepath)
            state_dict.update(sd)
    return state_dict


def remove_prefix(state_dict, prefix):
    if prefix[-1] != '.':
        prefix += '.'
    res_dict = OrderedDict()
    for k_ in state_dict.keys():
        res_dict[k_.replace(prefix, '')] = state_dict[k_]
    return res_dict


def load_checkpoint(file,
                    model: torch.nn.Module,
                    strict: bool = True,
                    preprocess_fn: Optional[Callable[[dict], dict]] = None,
                    **kwargs):
    """Loads training states from a checkpoint file.

    Args:
        file: a file-like object (has to implement read(), readline(), tell(), and seek()), or a string or os.PathLike
            object containing a file name.
        model (:class:`torch.nn.Module`): Model to load saved weights and buffers.
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Optimizer to recuperate.
        lr_scheduler (:class:`torch.optim.lr_scheduler._LRScheduler`, optional):
            lr_scheduler to recuperate, defaults to None.
        strict (bool, optional): Whether to strictly enforce that the keys in :attr:`state_dict`
            of the checkpoint match the names of parameters and buffers in model, defaults to True.

    Returns:
        int: The saved epoch number.

    Raises:
        RuntimeError: Raise error if the model/optimizer cannot successfully be recuperated
    """
    start = time()
    if gpc.get_local_rank(ParallelMode.MODEL) == 0:
        model_state = load_state_dict(file)
        if preprocess_fn:
            model_state = preprocess_fn(model_state)
    else:
        model_state = dict()
    dist.barrier()
    print(f'Load file time: {time()-start:.3f} s')
    # pipeline
    if is_using_pp():
        model_state = partition_pipeline_parallel_state_dict(model, model_state, **kwargs)
    if "prefix" in kwargs.keys():
        if kwargs['prefix'] != '':
            model_state = remove_prefix(model_state, kwargs["prefix"])

    model.load_state_dict(model_state, strict=strict)
    broadcast_model(model)

    return -1
