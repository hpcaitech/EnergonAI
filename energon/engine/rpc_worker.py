import os
import time
import torch
import inspect
import torch.distributed.rpc as rpc
import sys
from energon.core import global_context as gpc
from energon.context import ParallelMode
from .rpc_utils import remote_cls_method, sync_cls_method, async_cls_method
from .pipeline_wrapper import PipelineCommWrapper

class ReturnDict:
    def __init__(self):
        self.rd = dict()

    def enqueue(self, key, output):
        self.rd[key] = output

    def top(self, key):
        while key not in self.rd:
            time.sleep(0.001)
        output = self.rd.pop(key)
        return output


class RPCWorker:
    def __init__(self,
                 model_class,
                 model_config,
                 model_type,
                 dtype,
                 max_batch_size: int = 1) -> None:
        self.model_class = model_class
        self.model_config = model_config
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        self.WORKER_NAME = "wok{}"
        self.model = None  # call the model
        self.rank = gpc.get_local_rank(ParallelMode.GLOBAL)

        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')
        self._init_self()

        self.return_dict = ReturnDict()

    def _init_self(self):
        print("[INFO] init model in rank {}".format(self.rank))

        if self.dtype == torch.half:
            self.model = self.model_class(**self.model_config).cuda().half()
        else:
            self.model = self.model_class(**self.model_config).cuda()
            # print("Pass")
        self.model.eval()

        self.model = PipelineCommWrapper(model=self.model, max_batch_size=self.max_batch_size, dtype=self.dtype)

    def run(self, key, inputs):
        # print("key: {}".format(key), flush=True)
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')
        for k, v in inputs.items():
            if v is not None:
                inputs[k] = v.cuda()  # non_blocking=True

        if (gpc.is_initialized(ParallelMode.PIPELINE)) and (not gpc.is_last_rank(ParallelMode.PIPELINE)):
            self.model.run(key, inputs)
            return None
        else:
            output, cur_key = self.model.run(key, inputs)
            self.return_dict.enqueue(cur_key, output.cpu())
            return self.return_dict.top(key)

        return None
