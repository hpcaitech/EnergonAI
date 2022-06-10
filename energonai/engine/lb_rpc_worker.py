import time
import torch

from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.logging import get_dist_logger

from .lb_pipeline_wrapper import LbPipelineCommWrapper
from .pipeline_wrapper import PipelineCommWrapper
from .vit_pipeline_wrapper import ViTPipelineCommWrapper
from .auto_pipeline_wrapper import AutoPipelineCommWrapper



from energonai.pipelinable import split_transformer_into_partitions

from energonai.context import mcfg

logger = get_dist_logger('energonai')

pipe_wrapper = {
                'vit': ViTPipelineCommWrapper,
                'bert': PipelineCommWrapper,
                'gpt': PipelineCommWrapper,
                'auto': AutoPipelineCommWrapper,
                'lb': LbPipelineCommWrapper,
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

        self.WORKER_NAME = "wok{}"
        self.models = None    # call the model
        self.rank = gpc.get_local_rank(ParallelMode.GLOBAL)
        torch.cuda.set_device(f'cuda:{gpc.get_local_rank(ParallelMode.GLOBAL)}')

        # self.trt_sample = None
        self._init_self()
        self.return_dict = ReturnDict() 


    def _init_self(self):
        logger.info("Init model in rank {}".format(self.rank))
        self.models = self.model_class(**self.model_config)

        if self.dtype == torch.half:
            for model in self.models:
                model.cuda().half().eval()
        else:
            for model in self.models:
                model = model.cuda().eval()

        try:        
            self.models = pipe_wrapper['lb'](model=self.models, max_batch_size=self.max_batch_size, dtype=self.dtype)
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
