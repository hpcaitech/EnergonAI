import time
import torch

from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.logging import get_dist_logger

from .pipeline_wrapper import PipelineCommWrapper
from .vit_pipeline_wrapper import ViTPipelineCommWrapper
from .auto_pipeline_wrapper import AutoPipelineCommWrapper

from energonai.pipelinable import split_transformer_into_partitions

from energonai.context import MEATCONFIG

logger = get_dist_logger('energonai')

pipe_wrapper = {
                'vit': ViTPipelineCommWrapper,
                'bert': PipelineCommWrapper,
                'gpt': PipelineCommWrapper,
                'auto': AutoPipelineCommWrapper,
               }

pipe_split = {
                'bert': split_transformer_into_partitions,
                'gpt': split_transformer_into_partitions,
            }


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

    def __init__(self, model_class, model_config, model_type, dtype, max_batch_size: int = 1, auto_pp: bool = False) -> None:

        self.model_class = model_class
        self.model_config = model_config
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.model_type = model_type
        # self.auto_pp = auto_pp

        self.WORKER_NAME = "wok{}"
        self.model = None    # call the model
        self.rank = gpc.get_local_rank(ParallelMode.GLOBAL)
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')

        # self.trt_sample = None
        if auto_pp:
            self._auto_pp_init_model()
        else:
            self._init_self()
        self.return_dict = ReturnDict() 

    def _auto_pp_init_model(self):
        logger.info("Init automatic pipeline model in rank {}".format(self.rank))
        submodules = pipe_split[self.model_type](self.model_class)
        self.model = submodules.get_submodule(f'submod_{gpc.get_local_rank(ParallelMode.PIPELINE)}')
        del submodules
        self.model = pipe_wrapper['auto'](model=self.model, max_batch_size=self.max_batch_size, dtype=self.dtype)


    def _init_self(self):
        logger.info("Init model in rank {}".format(self.rank))

        if self.dtype == torch.half:
            self.model = self.model_class(**self.model_config).cuda().half()
        else:
            self.model = self.model_class(**self.model_config).cuda()
        
        self.model.eval()

        if MEATCONFIG['trt_sample'] is not None:
            try:
                logger.info('Import Torch2Trt')
                from torch2trt import torch2trt 
                from energonai.engine import trt_converter       
            except:
                logger.error("Installation Required, \n \
                    follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html \
                    and https://github.com/NVIDIA-AI-IOT/torch2trt")

        if MEATCONFIG['trt_sample'] is not None and gpc.get_world_size(ParallelMode.MODEL) > 1:
            logger.error("Tensor Parallelism does not support TensorRT convert")
        elif MEATCONFIG['trt_sample'] is not None and gpc.get_world_size(ParallelMode.MODEL) == 1:
            self.model = torch2trt(self.model, MEATCONFIG['trt_sample'])
            logger.info("TensorRT convert complete.")
        
        try:        
            self.model = pipe_wrapper[self.model_type](model=self.model, max_batch_size=self.max_batch_size, dtype=self.dtype)
        except:
            logger.error(f'Only {pipe_wrapper.keys()} pipeline wrapper are supported.')

    def run(self, key, inputs):
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')
        for k, v in inputs.items():
            if v is not None:
                inputs[k] = v.cuda()    # non_blocking=True

        if (gpc.is_initialized(ParallelMode.PIPELINE)) and (not gpc.is_last_rank(ParallelMode.PIPELINE)):
            self.model.run(key, inputs)
            return None
        else:
            output, cur_key = self.model.run(key, inputs)
            self.return_dict.enqueue(cur_key, output.cpu())
            return self.return_dict.top(key)

        # return None
