import torch
import importlib

try:
    energon_scale_mask = importlib.import_module("energon_scale_mask")
except ImportError:
    raise RuntimeError('energon_scale_mask requires cuda extensions')


def scale_mask_softmax(batch_size, batch_seq_len, head_num, src, seq_len_list):
    src = src.contiguous()
    dst = energon_scale_mask.scale_mask_softmax_wrapper(batch_size, batch_seq_len, head_num, src, seq_len_list)
    return dst