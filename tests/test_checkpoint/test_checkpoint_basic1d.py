#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pprint
from functools import partial

import energon.nn as col_nn
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from energon.context.parallel_mode import ParallelMode
from energon.core import global_context as gpc
from energon.initialize import launch
from energon.logging import disable_existing_loggers
from energon.utils import free_port, is_using_pp
from energon.utils.checkpointing import gather_pipeline_parallel_state_dict, load_checkpoint, save_checkpoint


def build_pipeline(model):
    from energon.builder.pipeline import partition_uniform

    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    depth = len(model)
    start, end = partition_uniform(depth, pipeline_size, 1)[pipeline_rank][0]
    layers = []
    for i in range(depth):
        if start <= i < end:
            layers.append(model[i])
        else:
            layers.append(nn.Identity())
    return nn.Sequential(*tuple(layers))


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-2)


def check_basic_1d(rank, world_size, port):
    # config = dict(
    #     parallel=dict(pipeline=dict(size=2), tensor=dict(size=4, mode="1d")),
    # )
    disable_existing_loggers()
    launch(pp_size=2, tp_size=2, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # m1 = nn.Sequential(col_nn.Embedding1D(20, 12), col_nn.Linear1D(12, 20), col_nn.Classifier1D(20, 3),
    #                    col_nn.Embedding1D(20, 12), col_nn.Linear1D(12, 20), col_nn.Classifier1D(20, 3))
    # m1 = nn.Sequential(col_nn.Embedding1D(20, 12), col_nn.Embedding1D(20, 12))
    # m1 = nn.Sequential(col_nn.Linear1D(4, 2), col_nn.Linear1D(2, 4))
    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        m1 = nn.Sequential(col_nn.Embedding1D(16, 4), col_nn.Classifier1D(8, 4), col_nn.Linear1D(4, 2),
                           col_nn.Dropout1D(),
                           nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity())
    else:
        m1 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
                           col_nn.Embedding1D(16, 4), col_nn.Classifier1D(8, 4), col_nn.Linear1D(4, 2),
                           col_nn.Dropout1D())
    for name, param in m1.named_parameters():
        print("RANK {}: {}, {}".format(gpc.get_global_rank(), name, param.size()))
    sd1 = m1.state_dict()
    # print(f"Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd1)}\n")
    save_checkpoint("test.pt", 0, m1)
    # m2 = nn.Sequential(col_nn.Embedding1D(20, 12), col_nn.Linear1D(12, 20), col_nn.Classifier1D(20, 3),
    #                    col_nn.Embedding1D(20, 12), col_nn.Linear1D(12, 20), col_nn.Classifier1D(20, 3))
    m2 = nn.Sequential(col_nn.Embedding1D(16, 4), col_nn.Classifier1D(8, 4), col_nn.Linear1D(4, 2), col_nn.Dropout1D(),
                       col_nn.Embedding1D(16, 4), col_nn.Classifier1D(8, 4), col_nn.Linear1D(4, 2), col_nn.Dropout1D())

    if is_using_pp():
        m2 = build_pipeline(m2)
    load_checkpoint("test.pt", m2)
    sd2 = m2.state_dict()
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        sd2 = gather_pipeline_parallel_state_dict(sd2)
    print(f"Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd2)}\n")

    if gpc.get_global_rank() == 0:
        for k, v in sd1.items():
            assert k in sd2
            check_equal(v.to(torch.device("cpu")), sd2[k].to(torch.device("cpu")))


@pytest.mark.dist
def test_checkpoint_1d():
    world_size = 4
    run_func = partial(check_basic_1d, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_checkpoint_1d()