#include <cuda_fp16.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP16_32(x)                                                       \
  AT_ASSERTM(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat16,     \
             "Datatype not implemented")

#define CHECK_FP16_32_INPUT(x)                                                 \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FP16_32(x)
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

template <typename T>
void ker_scale_mask_softmax_launcher(int batch_size, int batch_seq_len,
                                     int head_num, T *correlation,
                                     const int *real_seq_len);

torch::Tensor scale_mask_softmax_wrapper(int batch_size, int batch_seq_len,
                                         int head_num,
                                         torch::Tensor correlation,
                                         torch::Tensor real_seq_len) {
  CHECK_FP16_32_INPUT(correlation);
  CHECK_INPUT(real_seq_len);

  if (correlation.dtype() == torch::kFloat32) {
    ker_scale_mask_softmax_launcher<float>(batch_size, batch_seq_len, head_num,
                                           correlation.data_ptr<float>(),
                                           real_seq_len.data_ptr<int>());
  } else if (correlation.dtype() == torch::kFloat16) {
    ker_scale_mask_softmax_launcher<__half>(
        batch_size, batch_seq_len, head_num,
        (__half *)correlation.data_ptr<at::Half>(),
        real_seq_len.data_ptr<int>());
  }

  return correlation;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_mask_softmax_wrapper", &scale_mask_softmax_wrapper,
        "scale mask softmax fusion");
}
