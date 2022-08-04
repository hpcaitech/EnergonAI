#include "compat.h"
#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>


static const char *_cudaGetErrorEnum(cublasStatus_t error)
{ 
  switch (error)
  {
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

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x) \
  AT_ASSERTM(x.dtype() == torch::kFloat32, "Datatype not implemented")
#define CHECK_FP16(x) \
  AT_ASSERTM(x.dtype() == torch::kFloat16, "Datatype not implemented")
#define CHECK_FP16_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x);      \
  CHECK_FP16(x)
#define CHECK_FP32_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x);      \
  CHECK_FP32(x)
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " " + file + ":" + std::to_string(line) + " \n");
  }
}
#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)


torch::Tensor mlp_gemm(torch::Tensor input_tensor, torch::Tensor weights, int algo = CUBLAS_GEMM_DEFAULT)
{
  // cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
  // float time;
  // cudaEventCreate(&start1);
  // cudaEventCreate(&stop1);
  // cudaEventCreate(&start2);
  // cudaEventCreate(&stop2);
  // cudaEventCreate(&start3);
  // cudaEventCreate(&stop3);

  // cudaEventRecord(start1);
  CHECK_FP16_INPUT(input_tensor);
  CHECK_FP16_INPUT(weights);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  int batch_size = input_tensor.sizes()[0];
  // int m = input_tensor.sizes()[1];
  // int k = input_tensor.sizes()[2];
  // int n = weights.sizes()[1];
  int m = input_tensor.sizes()[1] * batch_size;
  int k = input_tensor.sizes()[2];
  int n = weights.sizes()[1];
  half h_alpha = (half)1.0f;
  half h_beta = (half)0.0f;
  auto options = torch::TensorOptions()
                     .dtype(input_tensor.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);

  // cudaEventRecord(start2);
  auto output = torch::zeros({batch_size, input_tensor.sizes()[1], n}, options);
  // cudaEventRecord(stop2);
  // cudaEventSynchronize(start2);
  // cudaEventSynchronize(stop2);
  // cudaEventElapsedTime(&time, start2, stop2);
  // printf("allocate:%fms ", time);

  // cudaEventRecord(start3);
  // check_cuda_error(
  //     cublasGemmStridedBatchedEx(
  //         this->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //         n, m, k,
  //         &this->h_alpha,
  //         weights.data_ptr(), CUDA_R_16F, n, 0,
  //         input_tensor.data_ptr(), CUDA_R_16F, k, m * k,
  //         &this->h_beta,
  //         output.data_ptr(), CUDA_R_16F, n, m * n,
  //         batch_size,
  //         CUBLAS_COMPUTE_16F,
  //         static_cast<cublasGemmAlgo_t>(algo)));

  check_cuda_error(
      cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &h_alpha,
          weights.data_ptr(), CUDA_R_16F, n,
          input_tensor.data_ptr(), CUDA_R_16F, k,
          &h_beta,
          output.data_ptr(), CUDA_R_16F, n,
          CUBLAS_COMPUTE_16F,
          static_cast<cublasGemmAlgo_t>(algo)));

  // cudaEventRecord(stop3);
  // cudaEventSynchronize(start3);
  // cudaEventSynchronize(stop3);
  // cudaEventElapsedTime(&time, start3, stop3);
  // printf("compute:%fms ", time);

  // cudaEventRecord(stop1);
  // cudaEventSynchronize(start1);
  // cudaEventSynchronize(stop1);
  // cudaEventElapsedTime(&time, start1, stop1);
  // printf("total:%fms\n", time);

  return output;
}

int get_start_algo()
{
  return CUBLAS_GEMM_DEFAULT;
}

int get_end_algo()
{
  return CUBLAS_GEMM_ALGO23;
}

int get_start_algo_t_op()
{
  return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}

int get_end_algo_t_op()
{
  return CUBLAS_GEMM_ALGO15_TENSOR_OP;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("mlp_gemm", &mlp_gemm, py::arg("input"), py::arg("param"), py::arg("algo") = (int)CUBLAS_GEMM_DEFAULT);
  m.def("get_start_algo", &get_start_algo);
  m.def("get_end_algo", &get_end_algo);
  m.def("get_start_algo_t_op", &get_start_algo_t_op);
  m.def("get_end_algo_t_op", &get_end_algo_t_op);
}