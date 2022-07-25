#include "compat.h"
#include <cassert>
#include <torch/extension.h>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>
// #include <cuda_bf16.h>
static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x)                                                          \
  AT_ASSERTM(x.dtype() == torch::kFloat32, "Datatype not implemented")
#define CHECK_FP16(x)                                                          \
  AT_ASSERTM(x.dtype() == torch::kFloat16, "Datatype not implemented")
#define CHECK_FP16_INPUT(x)                                                    \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FP16(x)
#define CHECK_FP32_INPUT(x)                                                    \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FP32(x)
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}
#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)




torch::Tensor func_linear(torch::Tensor input_tensor, torch::Tensor weights)
{
    CHECK_FP16_INPUT(input_tensor);
    CHECK_FP16_INPUT(weights);
    // CHECK_FP32_INPUT(input_tensor);
    // CHECK_FP32_INPUT(weights);

    float f_alpha = 1.0f;
    float f_beta =  0.0f;
    half h_alpha = (half)f_alpha;
    half h_beta = (half)f_beta;

    const void* alpha = reinterpret_cast<void*>(&h_alpha);
    const void* beta = reinterpret_cast<void*>(&h_beta);
    // const void* alpha = reinterpret_cast<void*>(&f_alpha);
    // const void* beta = reinterpret_cast<void*>(&f_beta);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    auto options = torch::TensorOptions()
                     .dtype(input_tensor.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);

    int batch_size = input_tensor.sizes()[0]; // 32
    int m = input_tensor.sizes()[1]; // 64
    int n = input_tensor.sizes()[2]; //12288
    int k = weights.sizes()[0]; // 49152
    auto output = torch::zeros({batch_size, m, k}, options);


    check_cuda_error(cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_T,
                 m,
                 k,
                 n,
                 alpha,
                 input_tensor.data_ptr(),
                 CUDA_R_16F,
                 n,
                 weights.data_ptr(),
                 CUDA_R_16F,
                 k,
                 beta,
                 output.data_ptr(),
                 CUDA_R_16F,
                 k,
                 CUBLAS_COMPUTE_16F ,
                 CUBLAS_GEMM_DEFAULT));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("func_linear", &func_linear,
        "cublas wrapper for linear layer, only fp16");
}