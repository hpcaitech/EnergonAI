import torch
import torch.nn as nn
from colossalai.nn import Linear1D_Col, Linear1D_Row
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.utils import is_using_pp


class BoringModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if is_using_pp():
            pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
            gather_output = pp_rank == gpc.get_world_size(ParallelMode.PIPELINE) - 1
        else:
            pp_rank = 0
            gather_output = True
        if pp_rank % 2 == 0:
            self.dense = Linear1D_Col(4, 4, gather_output=gather_output)
        else:
            self.dense = Linear1D_Row(4, 4)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.dense.weight.fill_(1.0)
            self.dense.bias.fill_(1.0)

    def forward(self, x):
        return self.dense(x)


def get_correct_output(x: torch.Tensor, pp_world_size: int) -> torch.Tensor:
    def step(t):
        return t * 4 + 1
    for _ in range(pp_world_size):
        x = step(x)
    return x
