// #include "ATen/ATen.h"
// #include "ATen/AccumulateType.h"
// #include "ATen/cuda/CUDAContext.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
void transpose_pad(const T *src, const int batch_size, const int max_seq_len,
                   const int *seq_len_list, const int head_num,
                   const int size_per_head, T *dst);

template <typename T>
void transpose_depad(const T *src, const int batch_size, const int max_seq_len,
                     const int *seq_len_list, const int head_num,
                     const int size_per_head, T *dst);

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x)                                                          \
  AT_ASSERTM(x.dtype() == torch::kFloat32, "Datatype not implemented")
#define CHECK_FP32_INPUT(x)                                                    \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FP32(x)
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor transpose_pad_wrapper(torch::Tensor src, int batch_size,
                                    int max_seq_len, torch::Tensor seq_len_list,
                                    int head_num, int size_per_head) {
  CHECK_INPUT(src);
  CHECK_INPUT(seq_len_list);

  auto options = torch::TensorOptions()
                     .dtype(src.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);
  auto dst =
      torch::zeros({batch_size, head_num, max_seq_len, size_per_head}, options);

  if (src.dtype() == torch::kFloat32) {
    transpose_pad(src.data_ptr<float>(), batch_size, max_seq_len,
                  seq_len_list.data_ptr<int>(), head_num, size_per_head,
                  dst.data_ptr<float>());
  } else {
    transpose_pad((half *)src.data_ptr(), batch_size, max_seq_len,
                  seq_len_list.data_ptr<int>(), head_num, size_per_head,
                  (half *)dst.data_ptr());
  }

  return dst;
}

torch::Tensor transpose_depad_wrapper(torch::Tensor src, int batch_size,
                                      int sum_seq, int max_seq_len,
                                      torch::Tensor seq_len_list, int head_num,
                                      int size_per_head) {
  CHECK_INPUT(src);
  CHECK_INPUT(seq_len_list);

  auto options = torch::TensorOptions()
                     .dtype(src.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);
  auto dst = torch::zeros({1, sum_seq, head_num, size_per_head}, options);
  // dst.contiguous();
  if (src.dtype() == torch::kFloat32) {
    transpose_depad(src.data_ptr<float>(), batch_size, max_seq_len,
                    seq_len_list.data_ptr<int>(), head_num, size_per_head,
                    dst.data_ptr<float>());
  } else {
    transpose_depad((half *)src.data_ptr(), batch_size, max_seq_len,
                    seq_len_list.data_ptr<int>(), head_num, size_per_head,
                    (half *)dst.data_ptr());
  }

  return dst;
}

// Faster

/* build offsets */

void build_sequence_length_padding_offset_kernelLauncher(
    const int *sequence_length, const int batch_size, const int max_seq_len,
    int *valid_word_num, int *tmp_mask_offset);

void ft_build_padding_offsets_wrapper(torch::Tensor sequence_length,
                                      int batch_size, int max_seq_len,
                                      torch::Tensor valid_word_num,
                                      torch::Tensor tmp_mask_offset) {
  CHECK_INPUT(sequence_length);
  CHECK_INPUT(valid_word_num);
  CHECK_INPUT(tmp_mask_offset);

  build_sequence_length_padding_offset_kernelLauncher(
      sequence_length.data_ptr<int>(), batch_size, max_seq_len,
      valid_word_num.data_ptr<int>(), tmp_mask_offset.data_ptr<int>());
}

/* remove padding from embedding layer to transformer blocks */
template <typename T>
void remove_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                   const int *tmp_mask_offset,
                                                   int *mask_offset,
                                                   const int m, const int n);

torch::Tensor ft_remove_padding_wrapper(torch::Tensor src,
                                        torch::Tensor tmp_mask_offset,
                                        torch::Tensor mask_offset,
                                        int valid_word_num, int hidden_dim) {
  CHECK_INPUT(src);
  CHECK_INPUT(tmp_mask_offset);
  CHECK_INPUT(mask_offset);

  auto options = torch::TensorOptions()
                     .dtype(src.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);
  auto tgt = torch::zeros({1, valid_word_num, hidden_dim}, options);

  if (src.dtype() == torch::kFloat32) {

    remove_sequence_length_padding_kernelLauncher<float>(
        (float *)src.data_ptr(), (float *)tgt.data_ptr(),
        (int *)tmp_mask_offset.data_ptr(), (int *)mask_offset.data_ptr(),
        valid_word_num, hidden_dim);
  } else {
    remove_sequence_length_padding_kernelLauncher<half>(
        (half *)src.data_ptr(), (half *)tgt.data_ptr(),
        (int *)tmp_mask_offset.data_ptr(), (int *)mask_offset.data_ptr(),
        valid_word_num, hidden_dim);
  }
  return tgt;
}

/* add padding from transformer blocks to final output*/
template <typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                    const int *mask_offset,
                                                    const int m, const int n);
torch::Tensor ft_rebuild_padding_wrapper(torch::Tensor src,
                                         torch::Tensor mask_offset,
                                         int valid_word_num, int hidden_dim,
                                         int batch_size, int max_seq_len) {
  CHECK_INPUT(src);
  CHECK_INPUT(mask_offset);

  auto options = torch::TensorOptions()
                     .dtype(src.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);
  auto tgt = torch::zeros({batch_size, max_seq_len, hidden_dim}, options);
  // auto tgt = torch::zeros_like(src);

  if (src.dtype() == torch::kFloat32) {
    rebuild_sequence_length_padding_kernelLauncher<float>(
        (float *)src.data_ptr(), (float *)tgt.data_ptr(),
        (int *)mask_offset.data_ptr(), valid_word_num, hidden_dim);
  } else {
    rebuild_sequence_length_padding_kernelLauncher<half>(
        (half *)src.data_ptr(), (half *)tgt.data_ptr(),
        (int *)mask_offset.data_ptr(), valid_word_num, hidden_dim);
  }

  return tgt;
}

/* FT transpose and remove padding */
template <typename T>
void transpose_rebuild_padding_kernelLauncher(
    T *Q, T *K, T *V, T *q_buf, T *k_buf, T *v_buf, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int valid_word_num, const int *mask_offset);

void ft_transpose_rebuild_padding_wrapper(torch::Tensor Q, torch::Tensor K,
                                          torch::Tensor V, torch::Tensor q_buf,
                                          torch::Tensor k_buf,
                                          torch::Tensor v_buf, int batch_size,
                                          int seq_len, int head_num,
                                          int size_per_head, int valid_word_num,
                                          torch::Tensor mask_offset) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(q_buf);
  CHECK_INPUT(k_buf);
  CHECK_INPUT(v_buf);
  CHECK_INPUT(mask_offset);

  if (Q.dtype() == torch::kFloat32) {
    transpose_rebuild_padding_kernelLauncher(
        (float *)Q.data_ptr(), (float *)K.data_ptr(), (float *)V.data_ptr(),
        (float *)q_buf.data_ptr(), (float *)k_buf.data_ptr(),
        (float *)v_buf.data_ptr(), batch_size, seq_len, head_num, size_per_head,
        valid_word_num, (int *)mask_offset.data_ptr());
  } else {
    transpose_rebuild_padding_kernelLauncher(
        (half *)Q.data_ptr(), (half *)K.data_ptr(), (half *)V.data_ptr(),
        (half *)q_buf.data_ptr(), (half *)k_buf.data_ptr(),
        (half *)v_buf.data_ptr(), batch_size, seq_len, head_num, size_per_head,
        valid_word_num, (int *)mask_offset.data_ptr());
  }
}

/* FT rebuild padding and transpose */
template <typename T>
void transpose_remove_padding_kernelLauncher(
    T *src, T *dst, const int valid_word_num, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int *mask_offset);

torch::Tensor ft_transpose_remove_padding_wrapper(
    torch::Tensor src, int valid_word_num, int batch_size, int seq_len,
    int head_num, int size_per_head, torch::Tensor mask_offset) {
  CHECK_INPUT(src);
  CHECK_INPUT(mask_offset);

  auto options = torch::TensorOptions()
                     .dtype(src.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);
  auto tgt =
      torch::zeros({1, valid_word_num, head_num * size_per_head}, options);

  if (src.dtype() == torch::kFloat32) {
    transpose_remove_padding_kernelLauncher(
        (float *)src.data_ptr(), (float *)tgt.data_ptr(), valid_word_num,
        batch_size, seq_len, head_num, size_per_head,
        (int *)mask_offset.data_ptr());
  } else {
    transpose_remove_padding_kernelLauncher(
        (half *)src.data_ptr(), (half *)tgt.data_ptr(), valid_word_num,
        batch_size, seq_len, head_num, size_per_head,
        (int *)mask_offset.data_ptr());
  }

  return tgt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose_pad_wrapper", &transpose_pad_wrapper,
        "Turbo Transpose and Padding");
  m.def("transpose_depad_wrapper", &transpose_depad_wrapper,
        "Turbo Transpose and Depadding");
  m.def("ft_build_padding_offsets_wrapper", &ft_build_padding_offsets_wrapper,
        "Faster build offsets");
  m.def("ft_remove_padding_wrapper", &ft_remove_padding_wrapper,
        "Faster remove padding");
  m.def("ft_rebuild_padding_wrapper", &ft_rebuild_padding_wrapper,
        "Faster rebuild padding");
  m.def("ft_transpose_remove_padding_wrapper",
        &ft_transpose_remove_padding_wrapper,
        "Faster transpose and remove padding");
  m.def("ft_transpose_rebuild_padding_wrapper",
        &ft_transpose_rebuild_padding_wrapper,
        "Faster transpose and rebuild padding");
}