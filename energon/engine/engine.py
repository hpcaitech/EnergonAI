import os
import time
import torch
from torch.nn import Module
import torch.multiprocessing as mp
from functools import partial
import requests

# pytorch rpc
import torch.distributed.rpc as rpc
from .rpc_utils import remote_cls_method, sync_cls_method, async_cls_method
from .rpc_worker import RPCWorker

from energon.core import global_context as gpc
from energon.context import ParallelMode
from energon.initialize import launch_from_torch, launch_from_multiprocess

from energon.utils import ensure_directory_exists
from energon.logging import get_dist_logger
from energon.nn import PipelineCommWrapper


class InferenceEngine(Module):
    def __init__(self, 
                model_class,
                model_config,               
                max_batch_size: int = 1,             
                tp_init_size: int = -1,
                pp_init_size: int = -1,
                host: str = 'localhost',
                port: int = 29500,
                dtype=None,
                checkpoint=None
                ):
        """
        Args:
            model: torch.nn.Module
            dtype: data-type by which inference is executed
            checkpoint: load parameter.
        """
        super().__init__()
        
        self.model_class = model_class
        self.model_config = model_config
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.max_batch_size = max_batch_size
             
        # for gpc
        self.rank = 0
        self.global_world_size = pp_init_size * tp_init_size
        self.host = host
        self.port = port
        self.tp_size = tp_init_size
        self.pp_size = pp_init_size

        # for TP
        self.rrefs = []        
        # for rpc
        self.WORKER_NAME = "wok{}"        
        self._init_dist_rpc()
        self._init_model()
    
    def _init_dist_rpc(self):
        r'''
        Based on global_context, init the rpc connection.
        '''
        os.environ['MASTER_ADDR'] = self.host
        os.environ['MASTER_PORT'] = f'{self.port}'        
        launch_from_multiprocess(tp_size = self.tp_size, pp_size = self.pp_size, rank = self.rank, local_rank = self.rank, world_size = self.global_world_size, host = self.host, port = self.port)
        rpc.init_rpc(self.WORKER_NAME.format(0), rank=0, world_size=self.global_world_size)
    
        
    def _init_model(self):
        for i in range(self.global_world_size):
            print(f'[INFO] rank{self.rank} calls rank{i} to init.')
            ob_info = rpc.get_worker_info(self.WORKER_NAME.format(i))
            self.rrefs.append(rpc.remote(ob_info, RPCWorker, args=(self.model_class, self.model_config, self.dtype, self.checkpoint, self.max_batch_size)))
        
    def run(self, inputs): 

        res_rref = 0
        output = None
        for rref in self.rrefs:
            output = remote_cls_method(RPCWorker.run, rref, inputs)
        
        return output
        

    def clear(self):
        
        rpc.shutdown()


        
