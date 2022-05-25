import torch
import importlib

try:
    energonai_transpose_pad = importlib.import_module("energonai_transpose_pad")
except ImportError:
    raise RuntimeError('transpose_pad requires cuda extensions')

# from transpose import transpose_pad_wrapper, transpose_depad_wrapper


def transpose_pad(src, batch_size, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = energonai_transpose_pad.transpose_pad_wrapper(src, batch_size, max_seq_len, seq_len_list, head_num,
                                                      size_per_head)

    return dst


def transpose_depad(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = energonai_transpose_pad.transpose_depad_wrapper(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num,
                                                        size_per_head)

    return dst


def depad(src, batch_size, seq_lens):
    dst = src[0:1, 0:seq_lens[0], :]

    for i in range(1, batch_size):
        tlen = seq_lens[i]
        dst = torch.cat([dst, src[i:i + 1, 0:tlen, :]], dim=1)

    return dst


# From FasterTransformer


def ft_build_padding_offsets(seq_lens, batch_size, max_seq_len, valid_word_num, tmp_mask_offset):
    seq_lens = seq_lens.contiguous()
    # tmp_mask_offset = tmp_mask_offset.contiguous()

    energonai_transpose_pad.ft_build_padding_offsets_wrapper(seq_lens, batch_size, max_seq_len, valid_word_num,
                                                           tmp_mask_offset)


def ft_remove_padding(src, tmp_mask_offset, mask_offset, valid_word_num, hidden_dim):
    src = src.contiguous()
    # tmp_mask_offset = tmp_mask_offset.contiguous()
    # mask_offset = mask_offset.contiguous()

    dst = energonai_transpose_pad.ft_remove_padding_wrapper(src, tmp_mask_offset, mask_offset, valid_word_num, hidden_dim)
    return dst


def ft_rebuild_padding(src, mask_offset, valid_word_num, hidden_dim, batch_size, max_seq_len):
    src = src.contiguous()
    # mask_offset = mask_offset.contiguous()

    dst = energonai_transpose_pad.ft_rebuild_padding_wrapper(src, mask_offset, valid_word_num, hidden_dim, batch_size,
                                                           max_seq_len)
    return dst


def ft_transpose_rebuild_padding(Q, K, V, q_buf, k_buf, v_buf, batch_size, seq_len, head_num, size_per_head,
                                 valid_word_num, mask_offset):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    q_buf = q_buf.contiguous()
    k_buf = k_buf.contiguous()
    v_buf = v_buf.contiguous()

    energonai_transpose_pad.ft_transpose_rebuild_padding_wrapper(Q, K, V, q_buf, k_buf, v_buf, batch_size, seq_len,
                                                               head_num, size_per_head, valid_word_num, mask_offset)


def ft_transpose_remove_padding(src, valid_word_num, batch_size, seq_len, head_num, size_per_head, mask_offset):
    src = src.contiguous()

    dst = energonai_transpose_pad.ft_transpose_remove_padding_wrapper(src, valid_word_num, batch_size, seq_len, head_num,
                                                                    size_per_head, mask_offset)
    return dst
