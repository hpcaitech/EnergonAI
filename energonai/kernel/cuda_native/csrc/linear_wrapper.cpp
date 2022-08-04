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
  // clock_t start, end, head;
  // head = clock();

  CHECK_FP16_INPUT(input_tensor);
  CHECK_FP16_INPUT(weights);
  // CHECK_FP32_INPUT(input_tensor);
  // CHECK_FP32_INPUT(weights);
  int batch_size = input_tensor.sizes()[0];
  int m = input_tensor.sizes()[1];
  int k = input_tensor.sizes()[2];
  int n = weights.sizes()[1];
  auto options = torch::TensorOptions()
                     .dtype(input_tensor.dtype())
                     .device(torch::kCUDA)
                     .requires_grad(false);

  // start = clock();
  auto output = torch::zeros({batch_size, m, n}, options);
  // end = clock();
  // std::cout << "allocate=" << double(end - start) / CLOCKS_PER_SEC << "s, ";

  // start = clock();
  check_cuda_error(
      cublasGemmStridedBatchedEx(
          this->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &this->h_alpha,
          weights.data_ptr(), CUDA_R_16F, n, 0,
          input_tensor.data_ptr(), CUDA_R_16F, k, m * k,
          &this->h_beta,
          output.data_ptr(), CUDA_R_16F, n, m * n,
          batch_size,
          CUBLAS_COMPUTE_16F,
          static_cast<cublasGemmAlgo_t>(algo)));
  // end = clock();
  // std::cout << "compute=" << double(end - start) / CLOCKS_PER_SEC << "s algo:" << algo << ", ";
  // std::cout << "all=" << double(end - head) / CLOCKS_PER_SEC << "s" << std::endl;

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