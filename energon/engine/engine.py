import os
import torch
from torch.nn import Module


# depend on colossalai
from energon.core import global_context as gpc
from energon.context import ParallelMode
from energon.initialize import launch_from_torch

from energon.utils import ensure_directory_exists
from energon.logging import get_dist_logger
from energon.nn import PipelineCommWrapper


class InferenceEngine(Module):
    def __init__(self, 
                model_class,
                model_config,
                samples,
                pp_init_size: int = -1,
                tp_init_size: int = -1,
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
        
        self._model_class = model_class
        self._model_config = model_config
        self._samples = samples
        self._pp_size = pp_init_size
        self._tp_size = tp_init_size
        self._dtype = dtype
        self._checkpoint = checkpoint
        self._model = None
        
        self._init_dist()
        self._set_sample_device()
        self._init_model()
        

        if self._checkpoint:
            # self._save_parameter()
            self._load_parameter()
        
    
    def _init_dist(self):
        launch_from_torch(tp_size = self._tp_size, pp_size = self._pp_size)
    
    def _set_sample_device(self):        
        for k, v in self._samples.items():
            if v is not None:
                self._samples[k] = v.cuda() 
    
    def _init_model(self):
        """
        TODO(dujiangsu) support other dtype 
        """  
        if self._dtype == torch.half:
            model = self._model_class(**self._model_config).cuda().half()
        else:
            model = self._model_class(**self._model_config).cuda()
        model.eval()
        self._model = PipelineCommWrapper(model = model, sample = self._samples, dtype=self._dtype)

    def _reinit_dist(self):
        gpc.destroy_vice_groups()
        config = dict(parallel = dict(pipeline=dict(size=self._pp_size),tensor=dict(size=self._tp_size, mode='1d')))
        gpc.load_config(config)
        gpc.init_parallel_groups()
    
        logger = get_dist_logger()
        logger.info(f'Switch is triggered and Distributed environment is re-initialized, '
                    f'pipeline parallel size: {gpc.pipeline_parallel_size}, '
                    f'tensor parallel size: {gpc.tensor_parallel_size}', ranks=[0])
    
    def _reload_model(self):
        del self._model
        self._init_model()

    def _get_ranks_name(self):
        # tensor parallel
        tp_local_rank = 0
        if gpc.is_initialized(ParallelMode.TENSOR):
            tp_local_rank = gpc.get_local_rank(ParallelMode.TENSOR)

        # pipeline parallel
        pp_local_rank = 0
        if gpc.is_initialized(ParallelMode.PIPELINE):
            pp_local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        ranks_name = f'tp{tp_local_rank}-pp{pp_local_rank}'
        return ranks_name

    def dtype_convert(self):     
        """
        TODO(dujiangsu) support other dtype 
        """  
        if self._dtype == torch.half:
            self._model.half()
        elif self._dtype == torch.float:
            self._model.float()
        
    
    def _save_parameter(self):
        """
        save checkpoint.
        """
        ensure_directory_exists(self._checkpoint)

        ranks_name = self._get_ranks_name()        
        ckpt_filename =  f'{ranks_name}.pt'

        checkpoint_path = os.path.join(self._checkpoint, ckpt_filename)
        # print(self._model)
        torch.save(self._model.state_dict(), checkpoint_path)


    def _repartition_weights(self):
        """
        TODO: The method can repartition weights among all devices based on new tp/pp strategy through communication.
        """

    def _load_parameter(self):  
        """
        TODO(dujiangsu) use json file to describe the distributed checkpoint. Like the strategy of Megatron.
        TODO(dujiangsu) based on the current tp/pp configuration, the func can re-partition the existing ckpt automatically.
            use self.repartition_weights() to avoid communication between host and device.
        """
        ensure_directory_exists(self._checkpoint)
        ranks_name = self._get_ranks_name()        
        ckpt_filename =  f'{ranks_name}.pt'
        checkpoint_path = os.path.join(self._checkpoint, ckpt_filename)

        self._model.load_state_dict(torch.load(checkpoint_path))

    def apply_new_parallel_strategy(self):
        """
        TODO: Switch between different tp/pp.
        """
        # decide new parallel strategy, re-create communication group.
        # repartition models and communicate weights.

        self.repartition_weights()
    
    def switch(self, pp_size, tp_size):
        """
        TP/PP switch trigger, triggered from remote.
        """
        self._pp_size = pp_size
        self._tp_size = tp_size
        self._reinit_dist()
        self._reload_model()
        
    def run(self):
        output = None
        with torch.inference_mode():
            output = self._model.run()        
        # if gpc.is_last_rank(ParallelMode.PIPELINE):
        return output

    