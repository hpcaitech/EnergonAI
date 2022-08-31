from torch import dtype, nn
from energonai.nn import Classifier1D
from colossalai.nn import VocabParallelClassifier1D
from energonai.nn.layer.parallel_1d._utils import gather_forward_split_backward
from colossalai.context import ParallelMode


class LMHead1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 word_embedding_weight: nn.Parameter = None,
                 bias: bool = False,
                 dtype: dtype = None,
                 vocab_parallel: bool = False) -> None:
        super().__init__()
        self.vocab_parallel = vocab_parallel
        if vocab_parallel:
            self.dense = VocabParallelClassifier1D(hidden_size, vocab_size, bias=bias, dtype=dtype)
        else:
            self.dense = Classifier1D(hidden_size, vocab_size, word_embedding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        if self.vocab_parallel:
            x = gather_forward_split_backward(x, ParallelMode.PARALLEL_1D, -1)
        return x
