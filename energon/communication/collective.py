#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch import Tensor

from energon.context import ParallelMode
from energon.core import global_context as gpc
from energon.utils import get_current_device


def all_gather(tensor: Tensor, dim: int, parallel_mode: ParallelMode, async_op: bool = False) -> Tensor:
    """Gathers all tensors from the parallel group and concatenates them in a 
    specific dimension.
    
    :param tensor: Tensor to be gathered
    :param dim: The dimension concatenating in
    :param parallel_mode: Parallel group mode used in this communication
    :param async_op: Whether operations are asynchronous

    :type tensor: :class:`torch.Tensor`
    :type dim: int
    :type parallel_mode: :class:`colossalai.context.ParallelMode`
    :type async_op: bool, optional

    :return: The tensor generated by all-gather
    :rtype: :class:`torch.Tensor`
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = [tensor]
        work = None
    else:
        shape = list(tensor.shape)
        shape[0], shape[dim] = shape[dim], shape[0]
        shape[0] *= depth
        out = torch.empty(shape, dtype=tensor.dtype, device=get_current_device())
        temp = list(torch.chunk(out, depth, dim=0))
        work = dist.all_gather(tensor_list=temp,
                               tensor=tensor.transpose(0, dim).contiguous(),
                               group=gpc.get_group(parallel_mode),
                               async_op=async_op)
        out = torch.transpose(out, 0, dim)
    if async_op:
        return out, work
    else:
        return out


def reduce_scatter(tensor: Tensor,
                   dim: int,
                   parallel_mode: ParallelMode,
                   op: ReduceOp = ReduceOp.SUM,
                   async_op: bool = False) -> Tensor:
    """Reduces all tensors then scatters it in a specific dimension to all 
    members in the parallel group.
    
    :param tensor: Tensor to be reduced and scattered
    :param dim: The dimension scattering in
    :param parallel_mode: Parallel group mode used in this communication
    :param op: The type of reduce operation
    :param async_op: Whether operations are asynchronous

    :type tensor: :class:`torch.Tensor`
    :type dim: int
    :type parallel_mode: :class:`colossalai.context.ParallelMode`
    :type op: ReduceOp, optional
    :type async_op: bool, optional

    :return: The tensor generated by reduce-scatter
    :rtype: :class:`Tensor`
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        temp = list(map(lambda x: x.contiguous(), torch.chunk(tensor, depth, dim=dim)))
        out = torch.empty(temp[0].shape, dtype=tensor.dtype, device=get_current_device())
        work = dist.reduce_scatter(output=out,
                                   input_list=temp,
                                   op=op,
                                   group=gpc.get_group(parallel_mode),
                                   async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def all_reduce(tensor: Tensor,
               parallel_mode: ParallelMode,
               op: ReduceOp = ReduceOp.SUM,
               async_op: bool = False) -> Tensor:
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        work = None
    else:
        work = dist.all_reduce(tensor.contiguous(), op=op, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return tensor, work
    else:
        return tensor


def broadcast(tensor: Tensor, src: int, parallel_mode: ParallelMode, async_op: bool = False):
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        work = None
    else:
        work = dist.broadcast(tensor.contiguous(), src=src, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return tensor, work
    else:
        return tensor


def reduce(tensor: Tensor, dst: int, parallel_mode: ParallelMode, op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        work = None
    else:
        work = dist.reduce(tensor.contiguous(), dst=dst, op=op, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return tensor, work
    else:
        return tensor


def scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None):
    r"""Modified from `torch.distributed.scatter_object_list <https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#scatter_object_list>` to fix issues
    """
    if dist._rank_not_in_group(group):
        return

    if not isinstance(scatter_object_output_list, list) or len(scatter_object_output_list) < 1:
        raise RuntimeError("Expected argument scatter_object_output_list to be a list of size at least 1.")

    # set tensor device to cuda if backend is nccl
    device = torch.cuda.current_device() if dist.get_backend(group) == 'nccl' else torch.device("cpu")

    my_rank = dist.get_rank()  # use global rank
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[dist.distributed_c10d._object_to_tensor(obj) for obj in scatter_object_input_list])
        tensor_list = list(map(lambda x: x.to(device), tensor_list))
        tensor_sizes = list(map(lambda x: x.to(device), tensor_sizes))

    # Src rank broadcasts the maximum tensor size. This is because all ranks are
    # expected to call into scatter() with equal-sized tensors.
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long).to(device)

    dist.broadcast(max_tensor_size, src=src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8).to(device)
    dist.scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,
        src=src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long).to(device)
    dist.scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,
        src=src,
        group=group,
    )

    output_tensor, obj_tensor_size = output_tensor.cpu(), obj_tensor_size.cpu()
    # Deserialize back to object
    scatter_object_output_list[0] = dist.distributed_c10d._tensor_to_object(output_tensor, obj_tensor_size)
