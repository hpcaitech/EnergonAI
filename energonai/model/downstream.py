from torch import dtype, nn
from colossalai.nn import Classifier1D, VocabParallelClassifier1D


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
            self.dense = VocabParallelClassifier1D(hidden_size, vocab_size, bias=bias, dtype=dtype, gather_output=True)
        else:
            self.dense = Classifier1D(hidden_size, vocab_size, word_embedding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x
