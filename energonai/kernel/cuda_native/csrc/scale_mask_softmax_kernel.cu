// LightSeq
// Copyright 2019 Bytedance Inc.

#include "common.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <random>

template <typename T>
__global__ void ker_scale_mask_softmax(T *correlation, const int *real_seq_len,
                                       const int batch_seq_len) {
  int query_token_pos = blockIdx.y % batch_seq_len;
  if (query_token_pos >= real_seq_len[blockIdx.x]) {
    return;
  }

  int mask = 0; // can see the token when mask=0
  if (threadIdx.x > query_token_pos || threadIdx.x >= batch_seq_len) {
    mask = 1; // Can only see the token on the left side of it
  }

  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * batch_seq_len + threadIdx.x;
  float val = threadIdx.x < batch_seq_len ? (float)correlation[idx]
                                          : CUDA_FLOAT_INF_NEG;
  float max_val = blockReduceMax<float>(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0)
    smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(val - smax);
  float rsum = blockReduceSum<float>(val);
  __shared__ float ssum;
  if (threadIdx.x == 0)
    ssum = rsum;
  __syncthreads();

  if (threadIdx.x < batch_seq_len)
    correlation[idx] = (T)(val / ssum);
}

template <typename T>
void ker_scale_mask_softmax_launcher(int batch_size, int batch_seq_len,
                                     int head_num, T *correlation,
                                     const int *real_seq_len) {
  int block_dim = batch_seq_len;
  if (batch_seq_len < 1024) {
    block_dim = (batch_seq_len + 31) >> 5;
    block_dim *= 32;
  }

  ker_scale_mask_softmax<T>
      <<<dim3(batch_size, head_num * batch_seq_len), block_dim>>>(
          correlation, real_seq_len, batch_seq_len);
}

template void ker_scale_mask_softmax_launcher<float>(int batch_size,
                                                     int batch_seq_len,
                                                     int head_num,
                                                     float *correlation,
                                                     const int *real_seq_len);

template void ker_scale_mask_softmax_launcher<__half>(int batch_size,
                                                      int batch_seq_len,
                                                      int head_num,
                                                      __half *correlation,
                                                      const int *real_seq_len);