import inspect
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Union

from energon.communication import send_forward, recv_forward, send_tensor_meta, recv_tensor_meta
from energon.context import ParallelMode
from energon.core import global_context as gpc


class PipelineCommWrapper:
    def __init__(self,
                 model: nn.Module,
                 sample: Union[dict, torch.utils.data.DataLoader] = None,
                 dtype=torch.float) -> None:
        # TODO (dujiangsu): to make sample capability for different types. Iteration, Tensor, and others.

        self.sample = None
        self.model = model
        self.dtype = dtype

        self.input_tensors = None
        self.output_tensors = None
        self.recv_tensor_shape = None
        self.input_parameters = []

        # only rank 0
        # if gpc.is_first_rank(ParallelMode.PIPELINE):
        if isinstance(sample, dict):
            self.sample = sample
        elif isinstance(sample, torch.utils.data.DataLoader):
            raise NotImplementedError

        self._init_input_parameters()
        # print(self.input_parameters)

        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            self._init_tensor_meta()

    def _init_input_parameters(self):
        sig = inspect.signature(self.model.forward)
        parameters = sig.parameters  # dict
        for name, param in parameters.items():
            self.input_parameters.append(name)

    def _build_samples(self):
        for param in self.input_parameters:
            if self.sample[param] is None:
                self.sample[param] = self.input_tensors

    def _init_tensor_meta(self):
        with torch.inference_mode():
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                # print(*self.sample)
                # print(type(self.sample[0]))
                output = self.model(**self.sample)
                send_tensor_meta(output)
                send_forward(output)
            elif gpc.is_last_rank(ParallelMode.PIPELINE):
                self.recv_tensor_shape = recv_tensor_meta(self.recv_tensor_shape)
                self.input_tensors = recv_forward(self.recv_tensor_shape, dtype=self.dtype)  # only a tensor now
                self._build_samples()
                output = self.model(**self.sample)
            else:
                self.recv_tensor_shape = recv_tensor_meta(self.recv_tensor_shape)
                self.input_tensors = recv_forward(self.recv_tensor_shape, dtype=self.dtype)  # only a tensor now
                self._build_samples()
                output = self.model(**self.sample)
                send_tensor_meta(output)
                send_forward(output)

    def run(self):
        if gpc.is_initialized(ParallelMode.PIPELINE):
            self.pipeline_run()
        else:
            self.no_pipeline_run()

    def no_pipeline_run(self):
        output = self.model(**self.sample)
        return output

    def pipeline_run(self):
        with torch.inference_mode():
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                output = self.model(**self.sample)
                send_forward(output)
                return None
            elif gpc.is_last_rank(ParallelMode.PIPELINE):
                self.input_tensors = recv_forward(self.recv_tensor_shape, dtype=self.dtype)  # only a tensor now
                self._build_samples()
                output = self.model(**self.sample)
                return output
            else:
                self.input_tensors = recv_forward(self.recv_tensor_shape, dtype=self.dtype)  # only a tensor now
                self._build_samples()
                output = self.model(**self.sample)
                send_forward(output)
                return None
