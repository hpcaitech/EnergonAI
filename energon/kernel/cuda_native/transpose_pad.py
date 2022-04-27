import torch
import importlib

try:
    energon_transpose_pad = importlib.import_module("energon_transpose_pad")
except ImportError:
    raise RuntimeError('transpose_pad requires cuda extensions')


# from transpose import transpose_pad_wrapper, transpose_depad_wrapper

def transpose_pad(src, batch_size, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = energon_transpose_pad.transpose_pad_wrapper(src, batch_size, max_seq_len, seq_len_list, head_num, size_per_head)

    return dst
    

def transpose_depad(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = energon_transpose_pad.transpose_depad_wrapper(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num, size_per_head)

    return dst


def depad(src, batch_size, seq_lens):
    dst=src[0:1,0:seq_lens[0],:]
    
    for i in range(1, batch_size):
        tlen = seq_lens[i]
        dst = torch.cat([dst, src[i:i+1,0:tlen,:]], dim=1)

    return dst


