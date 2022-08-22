import torch.fx
from energonai.context import MEATCONFIG

class EnergonTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        leaves = MEATCONFIG["LeafSet"] # set([BertTransformerLayer])
        return type(m) in leaves