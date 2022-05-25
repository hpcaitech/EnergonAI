from energonai.kernel import ft_build_padding_offsets, ft_remove_padding, ft_rebuild_padding, ft_transpose_remove_padding, ft_transpose_rebuild_padding
import torch
import pytest


seq_lens =  torch.tensor([24,127,31,65,24,127,31,65], dtype=torch.int).cuda()
batch_size = 8
max_padding_size = 128
head_size = 64
head_num = 12
hidden_size = head_num * head_size


def test_kernel():
    hidden_states_q = torch.rand(batch_size, max_padding_size, hidden_size).cuda()
    hidden_states_k = torch.rand(batch_size, max_padding_size, hidden_size).cuda()
    hidden_states_v = torch.rand(batch_size, max_padding_size, hidden_size).cuda()
    
    
    tmp_mask_offset = torch.zeros(batch_size, max_padding_size, dtype=torch.int).cuda()
    mask_offset = torch.zeros(batch_size, max_padding_size, dtype=torch.int).cuda()
    valid_word_num = torch.zeros(1, dtype=torch.int).cuda()

    ft_build_padding_offsets(seq_lens, batch_size, max_padding_size, valid_word_num, tmp_mask_offset)
    q = ft_remove_padding(hidden_states_q, tmp_mask_offset, mask_offset, valid_word_num[0].item(), hidden_size)
    k = ft_remove_padding(hidden_states_k, tmp_mask_offset, mask_offset, valid_word_num[0].item(), hidden_size)
    v = ft_remove_padding(hidden_states_v, tmp_mask_offset, mask_offset, valid_word_num[0].item(), hidden_size)
        
    new_qkv_shape = q.shape[:-1] + (head_num, head_size)
    
    q = q.view(new_qkv_shape)
    k = k.view(new_qkv_shape)
    v = v.view(new_qkv_shape)
    print(q.size())

    q_buf = torch.zeros(batch_size, head_num, max_padding_size, head_size).cuda()
    k_buf = torch.zeros(batch_size, head_num, max_padding_size, head_size).cuda()
    v_buf = torch.zeros(batch_size, head_num, max_padding_size, head_size).cuda()

    ft_transpose_rebuild_padding(q, k, v, q_buf, k_buf, v_buf, batch_size, max_padding_size, head_num, head_size, valid_word_num[0].item(), mask_offset)

    print(q_buf.size())

    q_buf = ft_transpose_remove_padding(v_buf, valid_word_num[0].item(), batch_size, max_padding_size, head_num, head_size, mask_offset)

    print(q_buf.size())

    q_buf = ft_rebuild_padding(q_buf, mask_offset, valid_word_num[0].item(), hidden_size, batch_size, max_padding_size)

    print(q_buf.size())





    # ft_transpose_remove_padding()
    
    



    

    # void ft_transpose_remove_padding_wrapper(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor q_buf, torch::Tensor k_buf, torch::Tensor v_buf,
    #                                       int batch_size, int seq_len, int head_num, int size_per_head, int valid_word_num, torch::Tensor mask_offset){


    # print(new_hidden_states.size())

    # def ft_remove_padding(src, tmp_mask_offset, mask_offset, valid_word_num, hidden_dim):
    # def ft_rebuild_padding(src, mask_offset, valid_word_num, hidden_dim):
    # def ft_transpose_remove_padding(Q, K, V, q_buf, k_buf, v_buf, batch_size, seq_len, head_num, size_per_head, valid_word_num, mask_offset):
    # def ft_transpose_rebuild_padding(src, valid_word_num, batch_size, seq_len, head_num, size_per_head, mask_offset):


if __name__ == '__main__':
    test_kernel()