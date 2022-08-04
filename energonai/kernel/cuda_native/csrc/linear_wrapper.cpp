#include "linear_wrapper.h"
#include <time.h>


EnergonLinear::EnergonLinear()
{
  cublasCreate(&(this->cublas_handle));
}

EnergonLinear::~EnergonLinear()
{
  cublasDestroy(this->cublas_handle);
}

torch::Tensor EnergonLinear::mlp_gemm(torch::Tensor input_tensor, torch::Tensor weights, int algo)
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

  int batch_size = input_tensor.sizes()[0];
  // int m = input_tensor.sizes()[1];
  // int k = input_tensor.sizes()[2];
  // int n = weights.sizes()[1];
  int m = input_tensor.sizes()[1] * batch_size;
  int k = input_tensor.sizes()[2];
  int n = weights.sizes()[1];
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
          this->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &this->h_alpha,
          weights.data_ptr(), CUDA_R_16F, n,
          input_tensor.data_ptr(), CUDA_R_16F, k,
          &this->h_beta,
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

int EnergonLinear::get_start_algo()
{
  return this->start_algo;
}

int EnergonLinear::get_end_algo()
{
  return this->end_algo;
}

int EnergonLinear::get_start_algo_t_op()
{
  return this->start_algo_t_op;
}

int EnergonLinear::get_end_algo_t_op()
{
  return this->end_algo_t_op;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  py::class_<EnergonLinear>(m, "EnergonLinear")
      .def(py::init<>())
      .def("mlp_gemm", &EnergonLinear::mlp_gemm, py::arg("tensor"), py::arg("tensor"), py::arg("int") = (int)CUBLAS_GEMM_DEFAULT)
      .def("get_start_algo", &EnergonLinear::get_start_algo)
      .def("get_end_algo", &EnergonLinear::get_end_algo)
      .def("get_start_algo_t_op", &EnergonLinear::get_start_algo_t_op)
      .def("get_end_algo_t_op", &EnergonLinear::get_end_algo_t_op);
}