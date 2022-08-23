import inspect
import threading

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Union

from energonai.communication import send_forward, recv_forward, send_tensor_meta, recv_tensor_meta
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from .pipeline_meta import PipelineMeta
from .pipeline_msg_dict import PipelineMsgDict, CircleInt    # PipelineMsgPriorityQueue


# The Wrapper is only for Transformer Model.
class PipelineCommWrapper:

    def __init__(self, model: nn.Module, max_batch_size: int = 1, dtype=torch.float) -> None:
        # TODO (dujiangsu): to make sample capability for different types. Iteration, Tensor, and others.
        self.model = model
        self.dtype = dtype

        self.tensor_dim = 0
        self.hidden_size = 0
        self.max_batch_size = max_batch_size

        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            input_ids = torch.randint(1, 10, (max_batch_size, 512), dtype=torch.int64).cuda()
            attention_mask = torch.randint(0, 1, (max_batch_size, 1, 512), dtype=torch.int64).cuda()
            hidden_states = None
            sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
            self._init_tensor_meta(sample)

        self.pipe_msg_queue = PipelineMsgDict()
        self.lock = threading.Lock()
        self.key = CircleInt()

    def _init_tensor_meta(self, sample):

        with torch.inference_mode():
            recv_tensor_shape = None
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                output = self.model(hidden_states=None,
                                    input_ids=sample['input_ids'],
                                    attention_mask=sample['attention_mask'])    # ([32, 512, 1600])
                send_tensor_meta(output)
                send_forward(output)
                self.tensor_dim = output.dim()
                self.hidden_size = output.size()[-1]
            elif gpc.is_last_rank(ParallelMode.PIPELINE):
                recv_tensor_shape = recv_tensor_meta(recv_tensor_shape)
                input_tensor = recv_forward(recv_tensor_shape, dtype=self.dtype)    # only a tensor now
                self.tensor_dim = input_tensor.dim()
                self.hidden_size = input_tensor.size()[-1]
            else:
                recv_tensor_shape = recv_tensor_meta(recv_tensor_shape)
                input_tensor = recv_forward(recv_tensor_shape, dtype=self.dtype)    # only a tensor now
                self.tensor_dim = input_tensor.dim()
                self.hidden_size = input_tensor.size()[-1]
                output = self.model(hidden_states=input_tensor,
                                    input_ids=input_tensor,
                                    attention_mask=sample['attention_mask'])
                send_tensor_meta(output)
                send_forward(output)

    def run(self, key, inputs):
        if gpc.is_initialized(ParallelMode.PIPELINE):
            return self.run_with_pp(key, inputs)
        else:
            return self.run_without_pp(key, inputs)

    def run_without_pp(self, key, inputs):
        pipe_meta = None
        self.pipe_msg_queue.enqueue(key, inputs, pipe_meta)

        self.lock.acquire()

        cur_key = self.key.val
        sample, pipe_meta = self.pipe_msg_queue.top(cur_key)
        self.key.addOne()
        with torch.inference_mode():
            if 'hidden_states' in sample:
                sample.pop('hidden_states')
            output = self.model(hidden_states=None, **sample)
        self.lock.release()

        return output, cur_key

    '''
    hidden_size : ([32, 512, 1600])
    For different model type, fill_meta_tensor is different
    '''

    def fill_meta_tensor(self, inputs, pipe_meta):
        if 'seq_lens' in inputs:
            pipe_meta.get_meta_tensor()[0] = 1
            pipe_meta.get_meta_tensor()[1] = 1
            pipe_meta.get_meta_tensor()[2] = torch.sum(inputs['seq_lens'])
        else:
            pipe_meta.get_meta_tensor()[0] = inputs['input_ids'].shape[0]
            pipe_meta.get_meta_tensor()[1] = inputs['input_ids'].shape[0]
            pipe_meta.get_meta_tensor()[2] = inputs['input_ids'].shape[1]

        pipe_meta.get_meta_tensor()[3] = self.hidden_size
        pipe_meta.update_meta()

    def run_with_pp(self, key, inputs):
        pipe_meta = PipelineMeta(self.tensor_dim, self.max_batch_size)
        self.fill_meta_tensor(inputs, pipe_meta)
        self.pipe_msg_queue.enqueue(key, inputs, pipe_meta)

        self.lock.acquire()
        cur_key = self.key.val
        sample, pipe_meta = self.pipe_msg_queue.top(cur_key)
        self.key.addOne()

        with torch.inference_mode():

            if gpc.is_first_rank(ParallelMode.PIPELINE):
                output = self.model(hidden_states=None,
                                    input_ids=sample['input_ids'],
                                    attention_mask=sample['attention_mask'],
                                    seq_lens=inputs['seq_lens'] if 'seq_lens' in inputs else None)

                send_forward(output)
                self.lock.release()
                return None

            if gpc.is_last_rank(ParallelMode.PIPELINE):

                # print(f'get_tensor_shapes:{pipe_meta.get_tensor_shapes()}')
                input_tensor = recv_forward(pipe_meta.get_tensor_shapes(), dtype=self.dtype)
                output = self.model(hidden_states=input_tensor,
                                    input_ids=sample['input_ids'],
                                    attention_mask=sample['attention_mask'],
                                    seq_lens=inputs['seq_lens'] if 'seq_lens' in inputs else None)
                self.lock.release()
                return output, cur_key

            else:

                input_tensor = recv_forward(pipe_meta.get_tensor_shapes(), dtype=self.dtype)
                output = self.model(hidden_states=input_tensor,
                                    input_ids=sample['input_ids'],
                                    attention_mask=sample['attention_mask'],
                                    seq_lens=inputs['seq_lens'] if 'seq_lens' in inputs else None)
                send_forward(output)
                self.lock.release()
                return None
