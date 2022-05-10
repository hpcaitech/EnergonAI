import re
from collections import OrderedDict
from itertools import chain

import torch
import torch.distributed as dist

from . import is_using_pp
from ..communication.collective import scatter_object_list
from ..context.parallel_mode import ParallelMode
from ..core import global_context as gpc

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'

__all__ = ["partition_tensor_parallel_state_dict",
           "load_checkpoint",
           "gather_tensor_parallel_state_dict",
           "save_checkpoint"]

name_map = {'ln_2': 'norm2',
            'c_attn': 'query_key_value',
            'attn.c_proj': 'attn.dense',
            'ln_1': 'norm1',
            'c_fc': 'dense_1',
            'mlp.c_proj': 'mlp.dense_2'}


def broadcast_state_dict(state_dict, parallel_mode):
    state_dict = [state_dict.copy() if isinstance(state_dict, dict) else state_dict]
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    dist.broadcast_object_list(state_dict, src=src_rank, group=gpc.get_cpu_group(parallel_mode))
    return state_dict[0]


def partition_tensor_parallel_state_dict(state_dict: OrderedDict,
                                         parallel_mode: ParallelMode,
                                         dims: dict = dict(),
                                         partition_states: dict = dict()):
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    depth = gpc.get_world_size(parallel_mode)

    if gpc.get_local_rank(parallel_mode) == 0:

        partitioned_state_list = [dict() for _ in range(depth)]

        for key in list(state_dict.keys()):
            param = state_dict.pop(key)
            dim = dims.get(key, 0)
            do_partition = partition_states.get(key, True)
            if do_partition:
                param = torch.chunk(param, depth, dim=dim)
            for i, p in enumerate(partitioned_state_list):
                p[key] = param[i] if do_partition else param

    else:
        partitioned_state_list = [None for _ in range(depth)]

    partitioned_state = [None]
    scatter_object_list(partitioned_state, partitioned_state_list, src=src_rank, group=gpc.get_cpu_group(parallel_mode))
    return partitioned_state[0]


def gather_tensor_parallel_state_dict(
        state_dict: OrderedDict,
        parallel_mode: ParallelMode,
        dims: dict = dict(),
        partition_states: dict = dict(),
        keep_vars: bool = False,
):
    dst_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    depth = gpc.get_world_size(parallel_mode)

    for key in list(state_dict.keys()):
        param = state_dict.pop(key)
        param = param if keep_vars else param.detach()
        dim = dims.get(key, 0)
        do_partition = partition_states.get(key, True)
        if do_partition:
            temp = param.transpose(0, dim).contiguous()
            gather_list = None
            if gpc.get_local_rank(parallel_mode) == 0:
                shape = list(param.shape)
                shape[0], shape[dim] = shape[dim], shape[0]
                shape[0] *= depth
                param = torch.empty(shape, dtype=param.dtype, device=param.device)
                gather_list = list(torch.chunk(param, depth, dim=0))
            dist.gather(temp, gather_list, dst=dst_rank, group=gpc.get_cpu_group(parallel_mode))
            param = torch.transpose(param, 0, dim)
        # update params in state_dict only on local rank 0
        if gpc.get_local_rank(parallel_mode) == 0:
            state_dict[key] = param

    return state_dict


def _send_state_dict(state_dict, dst, parallel_mode):
    state_tensor, state_size = dist.distributed_c10d._object_to_tensor(state_dict)
    dist.send(state_size, dst, group=gpc.get_cpu_group(parallel_mode))
    dist.send(state_tensor, dst, group=gpc.get_cpu_group(parallel_mode))


def _recv_state_dict(src, parallel_mode):
    state_size = torch.tensor([0], dtype=torch.long)
    dist.recv(state_size, src, group=gpc.get_cpu_group(parallel_mode))
    state_tensor = torch.empty(state_size.item(), dtype=torch.uint8)
    dist.recv(state_tensor, src, group=gpc.get_cpu_group(parallel_mode))
    state_dict = dist.distributed_c10d._tensor_to_object(state_tensor, state_size)
    return state_dict


def partition_pipeline_parallel_state_dict(model, state_dict, **kwargs):
    pipeline_state = OrderedDict()
    prefix = "" if "prefix" not in kwargs.keys() else kwargs["prefix"]
    if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        # receive all states from prev stage
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            state_dict = _recv_state_dict(gpc.get_prev_global_rank(ParallelMode.PIPELINE), ParallelMode.PIPELINE)
        # move states to output
        for name, _ in model.named_parameters(recurse=True, prefix=prefix):
            if name in state_dict:
                pipeline_state[name] = state_dict.pop(name)
        for name, _ in model.named_buffers(recurse=True, prefix=prefix):
            if name in state_dict:
                pipeline_state[name] = state_dict.pop(name)
        for name, _ in model.named_modules(prefix=prefix):
            extra_state_key = name + "." + _EXTRA_STATE_KEY_SUFFIX
            if extra_state_key in state_dict:
                pipeline_state[extra_state_key] = state_dict.pop(extra_state_key)
        # send rest states to next stage
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            _send_state_dict(state_dict, gpc.get_next_global_rank(ParallelMode.PIPELINE), ParallelMode.PIPELINE)

    return pipeline_state


def gather_pipeline_parallel_state_dict(state_dict):
    gathered_states = ([None for _ in range(gpc.get_world_size(ParallelMode.PIPELINE))]
                       if gpc.get_local_rank(ParallelMode.PIPELINE) == 0 else None)
    dist.gather_object(
        state_dict,
        gathered_states,
        dst=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[0],
        group=gpc.get_cpu_group(ParallelMode.PIPELINE),
    )

    state_dict = (OrderedDict(chain.from_iterable(state.items() for state in gathered_states))
                  if gpc.get_local_rank(ParallelMode.PIPELINE) == 0 else OrderedDict())

    return state_dict


def remove_prefix(state_dict, prefix):
    if prefix[-1] != '.':
        prefix += '.'
    res_dict = OrderedDict()
    for k_ in state_dict.keys():
        res_dict[k_.replace(prefix, '')] = state_dict[k_]
    return res_dict


def load_checkpoint(
        file,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        strict: bool = True, **kwargs
):
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
    state_dict = (torch.load(file, map_location=torch.device("cpu"))
                  if gpc.get_local_rank(ParallelMode.MODEL) == 0 else None)

    # model states
    if state_dict is not None:
        state_dict = processing_HF_GPT(state_dict)
    model_state = state_dict.pop("model") if state_dict is not None else dict()
    # pipeline
    if is_using_pp():
        model_state = partition_pipeline_parallel_state_dict(model, model_state, **kwargs)
    if "prefix" in kwargs.keys():
        if kwargs['prefix'] != '':
            model_state = remove_prefix(model_state, kwargs["prefix"])
    # print("Rank {}: {}".format(gpc.get_global_rank(), model_state))
    # print("+"*30)
    # print(model_state.keys())
    try:
        model.load_state_dict(model_state, strict=strict)
    except RuntimeError as e:
        error_msgs = str(e)
        if error_msgs.startswith("Error(s) in loading state_dict for "):
            error_msgs = error_msgs.split("\n\t")[1:]
            dst_rank = gpc.get_ranks_in_group(ParallelMode.MODEL)[0]
            all_error_msgs = [None for _ in range(gpc.get_world_size(ParallelMode.MODEL))]
            dist.gather_object(error_msgs, all_error_msgs, dst=dst_rank, group=gpc.get_cpu_group(ParallelMode.MODEL))
            if gpc.get_global_rank() == 0:
                all_error_msgs = list(chain.from_iterable(all_error_msgs))
                raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(all_error_msgs)))
        else:
            raise e

    # broadcast the rest states
    state_dict = broadcast_state_dict(state_dict, ParallelMode.MODEL)

    # # optimizer states
    # if optimizer is not None and 'optimizer' in state_dict:
    #     optimizer.load_state_dict(state_dict['optimizer'])

    # # lr scheduler states
    # if lr_scheduler is not None and 'lr_scheduler' in state_dict:
    #     lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    # last epoch
    last_epoch = state_dict.pop("epoch", -1)

    return last_epoch


def save_checkpoint(file,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    **kwargs):
    """Stores the checkpoint to disk. Saves all the training components' parameters or buffers, such as model, optimizer,
    lr_scheduler etc. into a checkpoint dictionary.

    Args:
        file: a file-like object (has to implement write and flush) or a string or os.PathLike object containing a
            file name.
        epoch (int): Epoch number (indicates how many epochs have you trained this model).
        model (:class:`torch.nn.Module`): Model to be saved.
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Optimizer to be saved.
        lr_scheduler (Union[:class:`torch.optim.lr_scheduler`, :class:`colossalai.nn.lr_scheduler`], optional):
            lr_scheduler to be saved, defaults to None.
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    """
    # ckpt container
    checkpoint = {"epoch": epoch}
    prefix = "" if "prefix" not in kwargs.keys() else kwargs["prefix"]
    model_state = model.state_dict(prefix=prefix)
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        model_state = gather_pipeline_parallel_state_dict(model_state)
    if "prefix" in kwargs.keys():
        kwargs.pop("prefix")
    if gpc.get_global_rank() == 0:
        checkpoint["model"] = model_state
        for key_ in model_state:
            print(key_, model_state[key_].size())
        # try:
        #     print("saving checkpoint: {}".format(checkpoint))
        # except Exception as e:
        #     print(e)
        # if optimizer is not None:
        #     checkpoint['optimizer'] = optimizer.state_dict()

        # if lr_scheduler is not None:
        #     checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        # print("before save")
        torch.save(checkpoint, file, **kwargs)
        # print("after save")


def judge_t(key_):
    key_words = ['attn.query_key_value.weight', 'mlp.dense_1.weight', 'mlp.dense_2.weight', 'attn.dense.weight']
    for word_ in key_words:
        if word_ in key_:
            return True
    return False


def processing_HF_GPT(state_dict: OrderedDict):
    new_dict = OrderedDict()
    for k_ in state_dict.keys():
        new_k = module_name_mapping(k_)
        if new_k == "":
            continue

        new_v = state_dict[k_]
        if judge_t(new_k):
            new_v = torch.transpose(new_v, 0, 1)
        if "attn.query_key_value.weight" in new_k:
            num_ = re.search(r"blocks\.blk_\d+?\.", new_k)
            if num_:
                prefix = num_.group()
            else:
                prefix = ''
            # print("prefix: {}".format(prefix))
            q_, k_, v_ = torch.chunk(new_v, 3, 0)
            # new_dict[prefix + "attn.query_.weight"] = torch.transpose(q_, 0, 1)
            # new_dict[prefix + "attn.key_.weight"] = torch.transpose(k_, 0, 1)
            # new_dict[prefix + "attn.value_.weight"] = torch.transpose(v_, 0, 1)
            new_dict[prefix + "attn.query_.weight"] = q_
            new_dict[prefix + "attn.key_.weight"] = k_
            new_dict[prefix + "attn.value_.weight"] = v_
        elif "attn.query_key_value.bias" in new_k:
            num_ = re.search(r"blocks\.blk_\d+?\.", new_k)
            if num_:
                prefix = num_.group()
            else:
                prefix = ''
            print("prefix: {}".format(prefix))
            q_, k_, v_ = torch.chunk(new_v, 3, 0)
            new_dict[prefix + "attn.query_.bias"] = q_
            new_dict[prefix + "attn.key_.bias"] = k_
            new_dict[prefix + "attn.value_.bias"] = v_
        else:
            new_dict[new_k] = new_v
    new_dict['head.dense.weight'] = new_dict['embed.word_embeddings.weight'].clone()
    # print("="*100)
    # print(new_dict.keys())
    return {"model": new_dict, "epoch": 0}


def id_map(matched):
    value = matched.group('value')
    return "blocks.blk_{}.".format(value)


def module_name_mapping(ori_name: str):
    if ori_name == 'wte.weight':
        return "embed.word_embeddings.weight"
    elif ori_name == 'wpe.weight':
        return "embed.position_embeddings.weight"
    elif "ln_f" in ori_name:
        return ori_name.replace('ln_f', 'norm')
    elif ".attn.bias" in ori_name:
        return ""
    else:
        res = re.sub(r"h\.(?P<value>\d+)?\.", id_map, ori_name)
        for k_ in name_map.keys():
            res = res.replace(k_, name_map[k_])
        return res
