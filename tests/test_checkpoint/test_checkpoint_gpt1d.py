#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pprint
from functools import partial

import colossalai.nn as col_nn
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from example.gpt.gpt import gpt2_small
from energonai.context.parallel_mode import ParallelMode
from energonai.engine import InferenceEngine
from example.gpt import *
from energonai.core import global_context as gpc
from energonai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.utils import free_port, is_using_pp
from energonai.utils.checkpointing import gather_pipeline_parallel_state_dict, load_checkpoint, save_checkpoint


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-2)


def check_gpt_1d(rank, world_size, port):
    disable_existing_loggers()
    launch(pp_size=2, tp_size=2, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # state_prefix = "rk_{}.".format(gpc.get_global_rank())
    # parameter_prefix = "rk_{}".format(gpc.get_global_rank())
    state_prefix = ''
    parameter_prefix = ''
    m1 = gpt2_small(vocab_size=50257)
    sd1 = m1.state_dict(prefix=state_prefix)
    for name, param in m1.named_parameters(prefix=parameter_prefix):
        print("RANK {}: {}, {}".format(gpc.get_global_rank(), name, param.size()))
    save_checkpoint("gpt_test.pt", 0, m1, prefix=state_prefix)
    print("Rank {} building second GPT".format(gpc.get_global_rank()))
    m2 = gpt2_small(checkpoint=True, checkpoint_path="gpt_test.pt", prefix=parameter_prefix, vocab_size=50257)
    sd2 = m2.state_dict(prefix=state_prefix)
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        sd2 = gather_pipeline_parallel_state_dict(sd2)
    # print("Rank {} : {}".format(gpc.get_global_rank(), sd2))
    print("Rank {} gather done".format(gpc.get_global_rank()))
    # print(f'Rank {gpc.get_global_rank()}:{pprint.pformat(sd2)}')
    if gpc.get_global_rank() == 0:
        for k, v in sd1.items():
            assert k in sd2
            check_equal(v.to(torch.device("cpu")), sd2[k].to(torch.device("cpu")))


def test_gpt():
    world_size = 4
    run_func = partial(check_gpt_1d, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_gpt()
