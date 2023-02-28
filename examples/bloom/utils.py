import torch
from torch import nn, Tensor
import torch.distributed as dist
import bitsandbytes as bnb
import torch.nn.functional as F
from typing import Optional, List
import time
import datetime
from torch.distributed.distributed_c10d import ReduceOp
import copy
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig, AutoModelForCausalLM

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Model Size: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

class Linear8bitTP(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        weight_data=None,
        index=None,
        bias_data=None
    ):
        super(Linear8bitTP, self).__init__(
            input_features, output_features, bias
        )
        self.state = bnb.MatmulLtState()
        self.index = index
        self.bias = bias_data
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.register_parameter("SCB", nn.Parameter(torch.empty(0), requires_grad=False))
        self.weight = weight_data
        

    def quant(self):  
        weight = self.weight.data.contiguous().half().to(self.rank)
        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
        delattr(self, "weight")
        setattr(self, "weight", nn.Parameter(CB, requires_grad=False))
        delattr(self, "SCB")
        setattr(self, "SCB", nn.Parameter(SCB, requires_grad=False))
        del weight
        self.weight.data = self.weight.data.to("cpu")
        self.SCB.data = self.SCB.data.to("cpu")

    def forward(self, x):
        self.state.is_training = self.training
        
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        self.state.CB = self.weight.data
        self.state.SCB = self.SCB.data
        
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        tensor_list = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, out)
        out = torch.cat(tensor_list, dim=2)
        del tensor_list
        if self.state.CxB is not None:
            del self.state.CxB
        
        return out

class LinearTP(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features:int, out_features:int, bias:bool=False,
                 weight_data=None, bias_data=None, device="meta", dtype=None,
                 use_int8:bool=True
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearTP, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.use_int8 = use_int8
        if use_int8:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.weight = weight_data
            self.bias = bias_data
        
    def forward(self, x):
        if self.use_int8 == True:
            x = x.chunk(self.world_size, dim=2)[self.rank]
            out = F.linear(x, self.weight, self.bias)
            dist.all_reduce(out, op=ReduceOp.SUM)
        else:
            out = F.linear(x, self.weight, self.bias)
            tensor_list = [torch.zeros_like(out) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, out)
            out = torch.cat(tensor_list, dim=2)
            del tensor_list
        return out

class EmbeddingTP(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weight: Optional[Tensor] = None,
    ) -> None:
        super(EmbeddingTP, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight,
        )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.weight = weight

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        
        tensor_list = [torch.zeros_like(emb) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, emb)
        emb = torch.cat(tensor_list, dim=2)
        del tensor_list
        return emb

@torch.no_grad()
def replace_tp_module(model : torch.nn.Module, 
                      threshold : float = 6.0, 
                      modules_to_not_convert : str = "lm_head",
                      use_int8: bool = True
                      ) -> torch.nn.Module:
    """replace_tp_module 
    Args:
        model (torch.nn.Module): a meta model
        threshold (float, optional): _description_. Defaults to 6.0.
        modules_to_not_convert (str, optional): the model names which shall not be shard+quant. Defaults to "lm_head" (for Bloom)
        use_int8(boll, optional): use int8 quantization. Defaults to True.
    Returns:
        torch.nn.Module: the meta model after quantization and tensor parallel sharding
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_tp_module(module, threshold, modules_to_not_convert, use_int8)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            if use_int8 == True:    
                model._modules[name] = Linear8bitTP(
                        input_features=module.in_features,
                        output_features=module.out_features,
                        threshold=6.0,
                        weight_data=module.weight,
                        bias_data=module.bias,
                )

            else:
                model._modules[name] = LinearTP(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias_data=module.bias,
                        weight_data=module.weight,
                        bias = module.bias is not None,
                        use_int8 = use_int8
                )
        
        elif isinstance(module, nn.Embedding):
            model._modules[name] = EmbeddingTP(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                weight=module.weight,
            )
        
        elif isinstance(module, nn.Linear) and name == 'lm_head':
            model._modules[name] = LinearTP(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
                )
            model._modules[name].weight = model._modules['transformer']._modules['word_embeddings'].weight
    return model

@torch.no_grad()
def get_tp_model(model : nn.Module,
                 rank : int,
                 world_size : int, 
                 use_int8 : bool = True)-> torch.nn.Module:
    """get_tp_model
    Shard a meta model for rank process.
    Args:
        model (torch.nn.Module): a meta model to be shard for TP.
        rank (int): the rank number of this process.
        world_size (int): the world size
        use_int8 (bool, optional): use int8 quantization. Defaults to True.
    Returns:
        torch.nn.Module: a sharded meta model ready for recieving data from rank0.
    """
    model = replace_tp_module(model, use_int8=use_int8)
    for name, module in model.named_modules():
        if isinstance(module, Linear8bitTP):
            bias_list = list(module.bias.data.chunk(world_size, dim=0))
            bias = bias_list[rank]
            
            weight_list = list(module.weight.data.chunk(world_size, dim=0))
            weight = weight_list[rank]
            SCB = torch.zeros_like(bias).to("meta")
            
            delattr(module, "weight")
            setattr(module, "weight", nn.Parameter(weight.to(torch.int8), requires_grad=False))
            
            delattr(module, "SCB")
            setattr(module, "SCB", nn.Parameter(SCB.to(torch.float32), requires_grad=False))
            
            delattr(module, "bias")
            setattr(module, "bias", nn.Parameter(bias))
            
            
        if isinstance(module, EmbeddingTP):   
            weight_list = list(module.weight.chunk(world_size, dim=1))
            delattr(module, 'weight')
            weight = nn.Parameter(weight_list[rank])
            setattr(module, 'weight', weight)
            
        if isinstance(module, LinearTP):
            if name == 'lm_head':
                delattr(module, 'weight')
                setattr(module, 'weight', model._modules['transformer']._modules['word_embeddings'].weight)
            else:
                bias_list = list(module.bias.data.chunk(world_size, dim=0))
                bias = bias_list[rank]
            
                weight_list = list(module.weight.data.chunk(world_size, dim=0))
                weight = weight_list[rank]
                
                delattr(module, "weight")
                setattr(module, "weight", nn.Parameter(weight, requires_grad=False))
            
                delattr(module, "bias")
                setattr(module, "bias", nn.Parameter(bias))
                
    return model

@torch.no_grad()
def get_tp_model_list(model : torch.nn.Module,
                      meta_model : torch.nn.Module,
                      world_size : int,
                      use_int8 : bool=True
                     ) -> List[torch.nn.Module]:
    """get_tp_model_list
    Materizate a `world_size` models for each process.
    Args:
        model (torch.nn.Module): a materialized model. It is sacrificed after the function finishes executing.
        meta_model (torch.nn.Module): a meta model with the same structure as the `model`.
        world_size (int): the world size
        use_int8 (bool, optional): use int8 quantization. Defaults to True.
    Returns:
        List[torch.nn.Module]: a list of materialized models after sharding and quantization.
    """
    model = replace_tp_module(model, use_int8=use_int8)
    
    model_list = []
    dist_meta_model = replace_tp_module(meta_model, use_int8=use_int8)
    for i in range(world_size):
        model_list.append(copy.deepcopy(dist_meta_model))
    
    # quantize and shard parameters
    for name, module in model.named_modules():
        if isinstance(module, Linear8bitTP):
            module.quant()
            weight_list = list(module.weight.data.chunk(world_size, dim=0))
            SCB_list = list(module.SCB.data.chunk(world_size, dim=0))
            bias_list = list(module.bias.data.chunk(world_size, dim=0))
            
            name_list = name.split('.')
            for rank in range(world_size):
                module_tmp = model_list[rank]._modules[name_list[0]]
                for i in range(1, len(name_list)):
                    module_tmp = module_tmp._modules[name_list[i]]
                delattr(module_tmp, "weight")
                setattr(module_tmp, "weight", nn.Parameter(weight_list[rank].clone().detach(), requires_grad=False))
                delattr(module_tmp, "SCB")
                setattr(module_tmp, "SCB", nn.Parameter(SCB_list[rank].clone().detach(), requires_grad=False))
                delattr(module_tmp, "bias")
                setattr(module_tmp, "bias", nn.Parameter(bias_list[rank].clone().detach()))
            del weight_list, SCB_list, bias_list, name_list
            module.to("meta")

        elif isinstance(module, EmbeddingTP):
            weight_list = list(module.weight.chunk(world_size, dim=1))
            name_list = name.split('.')
            for rank in range(world_size):
                module_tmp = model_list[rank]._modules[name_list[0]]
                for i in range(1, len(name_list)):
                    module_tmp = module_tmp._modules[name_list[i]]
                module_tmp.weight = nn.Parameter(weight_list[rank].clone().detach(), requires_grad=False)
            del name_list, weight_list
            module.to("meta")
            
        elif isinstance(module, LinearTP):
            if name == 'lm_head':
                name_list = name.split('.')
                for rank in range(world_size):
                    module_tmp = model_list[rank]._modules[name_list[0]]
                    for i in range(1, len(name_list)):
                        module_tmp = module_tmp._modules[name_list[i]]
                    module_tmp.weight = model_list[rank]._modules['transformer']._modules['word_embeddings'].weight
            else:
                name_list = name.split('.')
                weight_list = list(module.weight.data.chunk(world_size, dim=0))
                bias_list = list(module.bias.data.chunk(world_size, dim=0))
                for rank in range(world_size):
                    module_tmp = model_list[rank]._modules[name_list[0]]
                    for i in range(1, len(name_list)):
                        module_tmp = module_tmp._modules[name_list[i]]
                    delattr(module_tmp, "weight")
                    setattr(module_tmp, "weight", nn.Parameter(weight_list[rank].clone().detach(), requires_grad=False))
                    delattr(module_tmp, "bias")
                    setattr(module_tmp, "bias", nn.Parameter(bias_list[rank].clone().detach()))
                del name_list, weight_list, bias_list
            module.to("meta")
                
            
        elif len(list(module.children())) == 0:
            name_list = name.split('.')
            for rank in range(world_size):
                module_tmp = model_list[rank]._modules[name_list[0]]
                for i in range(1, len(name_list)):
                    module_tmp = module_tmp._modules[name_list[i]]
                try:
                    module_tmp.weight = module.weight
                except:
                    pass
                try:
                    module_tmp.bias = module.bias
                except:
                    pass
            module.to("meta")
    
    return model_list
            

from contextlib import contextmanager
@contextmanager
def init_empty_weights():
    old_register_parameter = nn.Module.register_parameter
    
    def register_empty_param(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)
            
    try:
        nn.Module.register_parameter = register_empty_param
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter

@contextmanager
def skip_init_context():
    old_init = nn.Linear.reset_parameters
    old_emb_init = nn.Embedding.reset_parameters
    
    def new_init(self):
        pass
    def new_emb_init(self):
        self._fill_padding_idx_with_zero()
        
    try:
        nn.Linear.reset_parameters = new_init
        nn.Embedding.reset_parameters = new_emb_init
        nn.LayerNorm.reset_parameters = new_init
        yield
    finally:
        nn.Linear.reset_parameters = old_init
        nn.Embedding.reset_parameters = old_emb_init
        
@contextmanager
def convert_param_attr_context(dtype=torch.float32, use_skip_init : bool = False):
    old_register_parameter = nn.Module.register_parameter
    
    def register_empty_param(module, name, param):
        if param is not None:
            param = nn.Parameter(param.data.to(dtype))
        old_register_parameter(module, name, param)
            
    try:
        nn.Module.register_parameter = register_empty_param
        if use_skip_init:
            with skip_init_context():
                yield
        else:
            yield
    finally:
        nn.Module.register_parameter = old_register_parameter


class ModelScatter(object):
    def __init__(self) -> None:
        self.cpu_group = dist.new_group(backend='gloo', timeout=datetime.timedelta(seconds=18000))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def _add_param(self, model, param_tensor, name):
        param = torch.nn.Parameter(param_tensor, requires_grad=False)
        name_list = name.split('.')
        module = model._modules[name_list[0]]
        for i in range(1, len(name_list) - 1):
            module = module._modules[name_list[i]]
        module._parameters[name_list[-1]] = param.to(self.rank)
        del param_tensor

    def scatter_model(self, src_model : torch.nn.Module, target_model : torch.nn.Module, use_int8 : bool = True) -> torch.nn.Module:
        """scatter_model

        Args:
            src_model (torch.nn.Module): a global materailized model
            target_model (torch.nn.Module): a meta model with the same structure as `src_model`
            use_int8(bool): use int8 quantization. Defaults to True
        
        Returns:
            torch.nn.Module: a local materailized model
        """
        if self.rank == 0:
            assert src_model.dtype == target_model.dtype, f"the src model and the target model should have the same dtype"
            assert src_model.device.type == 'cpu'
            assert target_model.device.type == 'meta'

            # get quant & sharded model_list
            time0 = time.time()
            model_list = get_tp_model_list(src_model, target_model, self.world_size, use_int8=use_int8)
            print("Model init complete", time.time() - time0)

            dist.barrier(self.cpu_group)
            # send out
            for name, param in model_list[0].named_parameters():
                param_list = [param.data]
                for i in range(1, self.world_size):
                    param_list.append(model_list[i].state_dict()[name])
                param_tensor = torch.zeros_like(
                    param_list[0], dtype=param_list[0].dtype)
                dist.scatter(param_tensor, scatter_list=param_list,
                                src=0, group=self.cpu_group)
                del param_list, param_tensor
            model = model_list[0]
            del model_list
            return model
        else:
            model = get_tp_model(target_model, self.rank, self.world_size, use_int8=use_int8)
            dist.barrier(self.cpu_group)
            for name, param in model.named_parameters():
                param_tensor = torch.zeros(
                    param.data.size(), dtype=param.dtype)
                dist.scatter(param_tensor, src=0, group=self.cpu_group)
                self._add_param(model, param_tensor, name)
            return model

def run_int8_bloom_inference(use_int8=True, from_pretrain=False, data_path=None, size="560m"):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_scatter = ModelScatter()

    if from_pretrain:
        configuration = BloomConfig.from_json_file(data_path + '/config.json')
    else:
        if size == "175b":
            configuration = BloomConfig(
                hidden_size=14336,
                n_layer=70,
                n_head=112,)
        elif size == "7b1":
            configuration = BloomConfig(
                hidden_size=4096,
                n_layer=30,
                n_head=32,)
        elif size == "560m":
            configuration = BloomConfig(
                hidden_size=1024,
                n_layer=24,
                n_head=16,)
        
    # meta init
    # get meta_model
    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(configuration).half()
    if rank == 0:           
        # get pre_trained model
        if from_pretrain:
            src_model = AutoModelForCausalLM.from_pretrained(
                data_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        else:
            with convert_param_attr_context(dtype=torch.float16, use_skip_init=True):
                src_model = AutoModelForCausalLM.from_config(configuration)
        print("src_model_load_complete")
            
        model = model_scatter.scatter_model(src_model, meta_model, use_int8)

    else:
        model = model_scatter.scatter_model(None, meta_model, use_int8)
        model._modules['lm_head']._parameters['weight'] = model._modules['transformer']._modules['word_embeddings'].weight
    
    getModelSize(model)
    return model

def run_fp16(from_pretrain=False, data_path=None):
    if from_pretrain:
        model = BloomForCausalLM.from_pretrained(
            data_path, low_cpu_mem_usage=True).half().to(0)
    else:
        cfg = BloomConfig(
            hidden_size=14336,
            n_layer=70,
            n_head=112,)
        with convert_param_attr_context(dtype=torch.float16, use_skip_init=True):
            model = BloomForCausalLM(cfg)
    
    return model

def run(tp=True, from_pretrain=False, data_path=None, use_int8=True, size="560m"):
    if tp:
        model = run_int8_bloom_inference(from_pretrain=from_pretrain, data_path=data_path, use_int8=use_int8, size=size)
    else:
        model = run_fp16(from_pretrain, data_path)
    return model
