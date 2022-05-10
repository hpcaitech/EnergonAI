#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

from energon.context import ParallelMode
from energon.core import global_context as gpc

__all__ = ['ParallelLayer']


class ParallelLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(
            ParallelMode.DATA)
        self.data_parallel_size = 1 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_world_size(
            ParallelMode.DATA)

        self.tensor_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_local_rank(
            ParallelMode.TENSOR)
        self.tensor_parallel_size = 1 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_world_size(
            ParallelMode.TENSOR)

        self.pipeline_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(
            ParallelMode.PIPELINE)
        self.pipeline_parallel_size = 1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(
            ParallelMode.PIPELINE)
