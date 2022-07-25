#include <cooperative_groups.h>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"
#include "one_layer_norm.cuh"
#include "type_shim.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void cuda_layer_norm(at::Tensor* output, at::Tensor* mean, at::Tensor* invvar, at::Tensor* input,
                     int n1, int n2, at::IntArrayRef normalized_shape, at::Tensor* gamma,
                     at::Tensor* beta, double epsilon) {
    at::Tensor normalized = at::empty_like(*output);
    fastfold::layer_norm::DirectLoad<at::BFloat16, float> load((at::BFloat16*)input->data_ptr(),
                                                               n2);
    fastfold::layer_norm::AffineStore<float, at::BFloat16, true, true> store(
        (at::BFloat16*)normalized.data_ptr(), (at::BFloat16*)output->data_ptr(), n2,
        (at::BFloat16*)gamma->data_ptr(), (at::BFloat16*)beta->data_ptr());
    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), float>(
        cuda_stream, load, store, n1, n2, epsilon, (float*)mean->data_ptr(),
        (float*)invvar->data_ptr());
}

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
    __device__ float* getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <typename T, typename U, typename V>
__device__ void cuLoadWriteStridedInputs(const int i1_block, const int thr_load_row_off,
                                         const int thr_load_col_off, const int i2_off,
                                         const int row_stride, U* warp_buf1, U* warp_buf2,
                                         const T* input, const V* dout, const int i1_end,
                                         const int n2, const U* __restrict__ mean,
                                         const U* __restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_mean = mean[i1];
        U curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf1[write_idx] = curr_dout;
                warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
            } else {
                warp_buf1[write_idx] = U(0);
                warp_buf2[write_idx] = U(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf1[write_idx] = U(0);
            warp_buf2[write_idx] = U(0);
        }
    }
}

template <typename T, typename U, typename V>
__device__ void cuLoadAddStridedInputs(const int i1_block, const int thr_load_row_off,
                                       const int thr_load_col_off, const int i2_off,
                                       const int row_stride, U* warp_buf1, U* warp_buf2,
                                       const T* input, const V* dout, const int i1_end,
                                       const int n2, const U* __restrict__ mean,
                                       const U* __restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_mean = mean[i1];
        U curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf1[write_idx] += curr_dout;
                warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
            }
        }
    }
}

template <typename T, typename U, typename V>
__global__ void cuComputePartGradGammaBeta(const V* __restrict__ dout, const T* __restrict__ input,
                                           const int n1, const int n2, const U* __restrict__ mean,
                                           const U* __restrict__ invvar, U epsilon,
                                           U* part_grad_gamma, U* part_grad_beta) {
    const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer();  // buf has at least blockDim.x * blockDim.y * blockDim.y +
                                   // (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
                             warp_buf1, warp_buf2, input, dout, i1_end, n2, mean, invvar);
    for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
         i1_block += blockDim.y * blockDim.y) {
        cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
                               warp_buf1, warp_buf2, input, dout, i1_end, n2, mean, invvar);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k * blockDim.y;
        int idx1 = row1 * row_stride + threadIdx.x;
        acc1 += warp_buf1[idx1];
        acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1 * row_stride + threadIdx.x;
            int idx2 = row2 * row_stride + threadIdx.x;
            warp_buf1[idx1] += warp_buf1[idx2];
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template <typename U, typename V>
__global__ void cuComputeGradGammaBeta(const U* part_grad_gamma, const U* part_grad_beta,
                                       const int part_size, const int n1, const int n2,
                                       V* grad_gamma, V* grad_beta) {
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        int num_warp_reductions = part_size / blockDim.y;
        U sum_gamma = U(0);
        U sum_beta = U(0);
        const U* part_grad_gamma_ptr =
            part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
        const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
        for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
            sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
            sum_beta += part_grad_beta_ptr[warp_offset * n2];
        }
        // inter-warp reductions
        const int nbsize3 = blockDim.x * blockDim.y / 2;
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
                buf[write_idx + nbsize3] = sum_beta;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
                sum_beta += buf[read_idx + nbsize3];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
            grad_beta[i2] = sum_beta;
        }
    }
}

template <typename T, typename U, typename V>
void HostLayerNormGradient(const V* dout, const U* mean, const U* invvar, at::Tensor* input, int n1,
                           int n2, const V* gamma, const V* beta, double epsilon, T* grad_input,
                           V* grad_gamma, V* grad_beta) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
        // compute grad_gamma(j) and grad_beta(j)
        const int part_size = 16;
        const dim3 threads2(32, 4, 1);
        const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
        const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
        const int nshared2_b = threads2.x * threads2.y * sizeof(U);
        const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
        at::Tensor part_grad_gamma =
            at::empty({part_size, n2}, input->options().dtype(at::ScalarType::Float));
        at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);
        cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
            dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, U(epsilon),
            part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(U);
        cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>(), part_size, n1, n2,
            grad_gamma, grad_beta);
    }
}

void cuda_layer_norm_gradient(at::Tensor* dout, at::Tensor* mean, at::Tensor* invvar,
                              at::Tensor* input, int n1, int n2, at::IntArrayRef normalized_shape,
                              at::Tensor* gamma, at::Tensor* beta, double epsilon,
                              at::Tensor* grad_input, at::Tensor* grad_gamma,
                              at::Tensor* grad_beta) {
    at::Tensor add_to_output = at::empty_like(*grad_input);
    fastfold::layer_norm::DirectLoad<at::BFloat16, float> load_x((at::BFloat16*)input->data_ptr(),
                                                                 n2);
    fastfold::layer_norm::ScaleLoad<at::BFloat16, float, true> load_scaled_dy(
        (at::BFloat16*)dout->data_ptr(), (at::BFloat16*)gamma->data_ptr(), n2);
    fastfold::layer_norm::AddStore<float, at::BFloat16, true> store(
        (at::BFloat16*)add_to_output.data_ptr(), (at::BFloat16*)grad_input->data_ptr(), n2);

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::layer_norm::DispatchLayerNormGrad<decltype(load_x), decltype(load_scaled_dy),
                                                decltype(store), float>(
        cuda_stream, load_x, load_scaled_dy, store, (float*)mean->data_ptr(),
        (float*)invvar->data_ptr(), n1, n2);

    using namespace at;
    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), gamma->scalar_type(), "cuda_layer_norm_gradient_kernel",
        HostLayerNormGradient(dout->DATA_PTR<scalar_t_out>(), mean->DATA_PTR<float>(),
                              invvar->DATA_PTR<float>(), input, n1, n2,
                              // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
                              // if gamma Tensor is NULL on input.
                              gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
                              gamma != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL, epsilon,
                              grad_input->DATA_PTR<scalar_t_in>(),
                              gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
                              gamma != NULL ? grad_beta->DATA_PTR<scalar_t_out>() : NULL);)
}