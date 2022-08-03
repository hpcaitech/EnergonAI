#pragma once

#include "compat.h"
#include <cassert>
#include <torch/extension.h>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>

class EnergonLinear
{
public:
    EnergonLinear();
    ~EnergonLinear();
    torch::Tensor mlp_gemm(torch::Tensor input_tensor, torch::Tensor weights, int algo = CUBLAS_GEMM_DEFAULT);
    int get_start_algo();
    int get_end_algo();
    int get_start_algo_t_op();
    int get_end_algo_t_op();

private:
    cublasHandle_t cublas_handle;

    half h_alpha = (half)1.0f;
    half h_beta = (half)0.0f;

    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
};

// #include <cuda_bf16.h>
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
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x);    \
    CHECK_FP16(x)
#define CHECK_FP32_INPUT(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x);    \
    CHECK_FP32(x)
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
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
