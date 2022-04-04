import os
import torch
import inspect
import torch.distributed.rpc as rpc
import sys
from energon.core import global_context as gpc
from energon.context import ParallelMode
from .rpc_utils import remote_cls_method, sync_cls_method, async_cls_method
from .gpt_pipeline_wrapper import GPTPipelineCommWrapper

class RPCWorker:
    def __init__(self, 
                 model_class, 
                 model_config,
                 dtype,
                 checkpoint,
                 max_batch_size:int  = 1) -> None:
        self.model_class = model_class
        self.model_config = model_config
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.max_batch_size = max_batch_size

        self.WORKER_NAME = "wok{}"
        self.model = None # call the model
        self.rank = gpc.get_local_rank(ParallelMode.GLOBAL)
        
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')    
        
        self._init_self()

    def _init_self(self):
        print("[INFO] init model in rank {}".format(self.rank))

        if self.dtype == torch.half:
            self.model = self.model_class(**self.model_config).cuda().half()
        else:
            self.model = self.model_class(**self.model_config).cuda()      
        # print("Pass")
        self.model.eval()


        if gpc.is_initialized(ParallelMode.PIPELINE):
            self.model = GPTPipelineCommWrapper(model = self.model, max_batch_size = self.max_batch_size, dtype=self.dtype)  

    def run(self, inputs):
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}') 
        for k, v in inputs.items():
            if v is not None:
                inputs[k] = v.cuda() #non_blocking=True

        if gpc.is_initialized(ParallelMode.PIPELINE):
            output = self.model.run(inputs)
        else:
            output = self.model(**inputs)
        
        if output is not None:
            return output.cpu() #non_blocking=True
        
        return None