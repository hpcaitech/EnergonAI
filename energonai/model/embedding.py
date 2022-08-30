import torch
from torch import nn as nn
from torch import dtype
from energonai.nn import VocabParallelEmbedding1D
from energonai.utils import get_current_device


class Embedding1D(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 max_seq_len: int,
                 num_tokentypes: int = 0,
                 padding_idx: int = 0,
                 dtype: dtype = None,
                 vocab_parallel: bool = False,
                 ) -> None:
        super().__init__()

        self.word_embeddings = VocabParallelEmbedding1D(
            vocab_size, hidden_size, padding_idx=padding_idx, dtype=dtype, skip_tp=not vocab_parallel)

        self.position_embeddings = VocabParallelEmbedding1D(max_seq_len, hidden_size, dtype=dtype, skip_tp=True)

        if num_tokentypes > 0:
            self.tokentype_embeddings = VocabParallelEmbedding1D(num_tokentypes, hidden_size, dtype=dtype, skip_tp=True)
        else:
            self.tokentype_embeddings = None

        # self.position_ids = torch.arange(max_seq_len, dtype=torch.long, device=get_current_device()).expand((1, -1))

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self,
                input_ids,
                position_ids=None,
                tokentype_ids=None,
                past_key_values_length: int = 0):

        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=get_current_device()).unsqueeze(0)
            # position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)

        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)

        return x
