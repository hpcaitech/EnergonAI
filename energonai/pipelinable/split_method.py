from torch.fx.passes.split_module import split_module
from .split_policy import module_equal_partition, naive_equal_partition, transformer_partition
from .energon_tracer import EnergonTracer
import torch.fx

def filter_graph(traced: torch.fx.GraphModule, filter_type: str):
    len = 0
    for node in traced.graph.nodes:
        if node.op == filter_type:
            len = len + 1
    return len
    

def split_transformer_into_partitions(model_class):
    model = model_class()
    graph = EnergonTracer().trace(model)
    traced = torch.fx.GraphModule(model, graph)
    depth = filter_graph(traced, "call_module") - 1
    submodules = split_module(traced, model, transformer_partition(depth))
    del model

    return submodules