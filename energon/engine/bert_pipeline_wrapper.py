import inspect
import threading

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Union

from energon.communication import send_forward, recv_forward, send_tensor_meta, recv_tensor_meta
from energon.context import ParallelMode
from energon.core import global_context as gpc

from .pipeline_meta import PipelineMeta
from .pipeline_msg_dict import PipelineMsgDict, CircleInt # PipelineMsgPriorityQueue, 


# The Wrapper is only for Transformer Model.
class BertPipelineCommWrapper:
    def __init__(self, 
                 model: nn.Module, 
                 max_batch_size: int = 1,
                 dtype = torch.float) -> None:
        # TODO (dujiangsu): to make sample capability for different types. Iteration, Tensor, and others.
        self.model = model
        self.dtype = dtype

        # input
        self.static_input = dict()
        self.static_name = []
        self.comm_input = dict()
        self.comm_name = None        

        # get the hidden_size
        input_ids = torch.randint(1, 10, (max_batch_size, 512), dtype=torch.int64).cuda()
        attention_mask = torch.randint(0, 1, (max_batch_size, 1, 512, 512), dtype=torch.int64).cuda()
        hidden_states = None
        self.sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
        self._init_input()
        self.tensor_dim = 0
        self.hidden_size = 0
        self.max_batch_size = max_batch_size

        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            self._init_tensor_meta()

        self.pipe_msg_queue = PipelineMsgDict() #PipelineMsgPriorityQueue()
        self.lock = threading.Lock()
        self.key = CircleInt()

    def _init_input(self):

        sig = inspect.signature(self.model.forward)
        parameters = sig.parameters # dict
        for name, _ in parameters.items():
            if self.sample[name] is not None:
                self.static_input[name] = self.sample[name]
                self.static_name.append(name)
            else:
                self.comm_input[name] = None
                self.comm_name = name


    def _init_tensor_meta(self):   
        
        with torch.inference_mode():
            recv_tensor_shape = None
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                output = self.model(**self.comm_input, **self.static_input) # ([32, 512, 1600])
                send_tensor_meta(output)
                send_forward(output)
                self.tensor_dim = output.dim()
                self.hidden_size = output.size()[-1]
            elif gpc.is_last_rank(ParallelMode.PIPELINE):                
                recv_tensor_shape = recv_tensor_meta(recv_tensor_shape)
                input_tensor = recv_forward(recv_tensor_shape, dtype=self.dtype) # only a tensor now
                self.tensor_dim = input_tensor.dim()
                self.hidden_size = input_tensor.size()[-1]        
            else:
                recv_tensor_shape = recv_tensor_meta(recv_tensor_shape)
                input_tensor = recv_forward(recv_tensor_shape, dtype=self.dtype) # only a tensor now
                self.tensor_dim = input_tensor.dim()
                self.hidden_size = input_tensor.size()[-1]
                self.comm_input[self.comm_name] = input_tensor
                output = self.model(**self.comm_input, **self.static_input)
                send_tensor_meta(output)
                send_forward(output)

    '''
    hidden_size : ([32, 512, 1600])
    For different model type, fill_meta_tensor is different
    '''
    def fill_meta_tensor(self, inputs, pipe_meta):
        pipe_meta.get_meta_tensor()[0] = inputs['input_ids'].shape[0]
        pipe_meta.get_meta_tensor()[1] = inputs['input_ids'].shape[0]
        pipe_meta.get_meta_tensor()[2] = inputs['input_ids'].shape[1]
        pipe_meta.get_meta_tensor()[3] = self.hidden_size
        pipe_meta.update_meta()

    def run(self, key, inputs): 

        print(f'Rank: {gpc.get_global_rank()}, Priority: {key}')

        pipe_meta = PipelineMeta(self.tensor_dim, self.max_batch_size)
        self.fill_meta_tensor(inputs, pipe_meta)
        self.pipe_msg_queue.enqueue(key, inputs, pipe_meta)

        # different threads ask for a single lock  how to keep the correctness
        self.lock.acquire(timeout=3)

        sample, pipe_meta = self.pipe_msg_queue.top(self.key.val)
        self.key.addOne()

        for name in self.static_name:
            self.static_input[name] = sample[name]

        with torch.inference_mode():

            if gpc.is_first_rank(ParallelMode.PIPELINE):

                output = self.model(**self.comm_input, **self.static_input)
                send_forward(output)
                self.lock.release()
                return None

            if gpc.is_last_rank(ParallelMode.PIPELINE):

                # print(f'get_tensor_shapes:{pipe_meta.get_tensor_shapes()}')
                input_tensor = recv_forward(pipe_meta.get_tensor_shapes(), dtype=self.dtype)
                # print(f'input_tensor.shape:{input_tensor.shape}')
                self.comm_input[self.comm_name] = input_tensor
                output = self.model(**self.comm_input, **self.static_input)
                self.lock.release()
                return output

            else:
                
                input_tensor = recv_forward(pipe_meta.get_tensor_shapes(), dtype=self.dtype)
                self.comm_input[self.comm_name] = input_tensor
                output = self.model(**self.comm_input, **self.static_input)
                send_forward(output)
                self.lock.release()
                return None

        