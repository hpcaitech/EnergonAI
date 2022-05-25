// For eliminating redundant computation.
// transpose and padding/depadding fusion to reduce the memory move.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// from turbo
template <typename T>
__global__ void transpose_depad_kernel(const T *src, const int batch_size,
                                       const int seq_len, const int *seq_lens,
                                       const int head_num,
                                       const int size_per_head, T *dst) {

  int idx = threadIdx.x;
  int batch_index = blockIdx.x / (head_num * seq_len);
  int head_index = (blockIdx.x % (head_num * seq_len)) / seq_len;
  int seq_index = blockIdx.x % seq_len;

  if (seq_index >= seq_lens[batch_index]) {
    return;
  }

  // to know the start place of each batch
  int sum_len = 0;
  for (size_t i = 0; i < batch_index; ++i) {
    sum_len += seq_lens[i];
  }
  while (idx < size_per_head) {
    // set the invalid elements to 0.
    dst[(sum_len + seq_index) * (head_num * size_per_head) +
        head_index * size_per_head + idx] =
        src[blockIdx.x * size_per_head + idx];
    idx += blockDim.x;
  }
}

template <typename T>
void transpose_depad(const T *src, const int batch_size, const int seq_len,
                     const int *seq_lens, const int head_num,
                     const int size_per_head, T *dst) {
  dim3 dimGrid(batch_size * head_num * seq_len);
  dim3 dimBlock(size_per_head);

  transpose_depad_kernel<<<dimGrid, dimBlock>>>(
      src, batch_size, seq_len, seq_lens, head_num, size_per_head, dst);
}

template void transpose_depad(const float *src, const int batch_size,
                              const int seq_len, const int *seq_lens,
                              const int head_num, const int size_per_head,
                              float *dst);

template void transpose_depad(const half *src, const int batch_size,
                              const int seq_len, const int *seq_lens,
                              const int head_num, const int size_per_head,
                              half *dst);

template <typename T>
__global__ void transpose_pad_kernel(const T *src, const int batch_size,
                                     const int seq_len, const int *seq_lens,
                                     const int head_num,
                                     const int size_per_head, T *dst) {

  int idx = threadIdx.x;
  int batch_index = blockIdx.x / (head_num * seq_len);
  int head_index = (blockIdx.x % (head_num * seq_len)) / seq_len;
  int seq_index = blockIdx.x % seq_len;

  // to know the start place of each batch
  int sum_len = 0;
  for (size_t i = 0; i < batch_index; ++i) {
    sum_len += seq_lens[i];
  }
  while (idx < size_per_head) {
    if (seq_index >= seq_lens[batch_index]) {
      dst[blockIdx.x * size_per_head + idx] = 0.f;
    } else {
      dst[blockIdx.x * size_per_head + idx] =
          src[(sum_len + seq_index) * (head_num * size_per_head) +
              head_index * size_per_head + idx];
    }
    idx += blockDim.x;
  }
}

template <typename T>
void transpose_pad(const T *src, const int batch_size, const int seq_len,
                   const int *seq_lens, const int head_num,
                   const int size_per_head, T *dst) {

  dim3 dimGrid(batch_size * head_num * seq_len);
  dim3 dimBlock(size_per_head);

  transpose_pad_kernel<<<dimGrid, dimBlock>>>(
      src, batch_size, seq_len, seq_lens, head_num, size_per_head, dst);
}

template void transpose_pad(const float *src, const int batch_size,
                            const int seq_len, const int *seq_lens,
                            const int head_num, const int size_per_head,
                            float *dst);

template void transpose_pad(const half *src, const int batch_size,
                            const int seq_len, const int *seq_lens,
                            const int head_num, const int size_per_head,
                            half *dst);

// from faster

/* create offsets */

__global__ void build_sequence_length_padding_offset(const int *sequence_length,
                                                     const int batch_size,
                                                     const int max_seq_len,
                                                     int *valid_word_num,
                                                     int *tmp_mask_offset) {
  // do cumulated sum
  int total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  for (int i = 0; i < batch_size; i++) {
    const int seq_len = sequence_length[i];
    for (int j = 0; j < seq_len; j++) {
      tmp_mask_offset[index] = cum_offset;
      index++;
    }
    cum_offset += max_seq_len - seq_len;
    total_seq_len += seq_len;
  }
  valid_word_num[0] = total_seq_len;
}

void build_sequence_length_padding_offset_kernelLauncher(
    const int *sequence_length, const int batch_size, const int max_seq_len,
    int *valid_word_num, int *tmp_mask_offset) {
  build_sequence_length_padding_offset<<<1, 1>>>(sequence_length, batch_size,
                                                 max_seq_len, valid_word_num,
                                                 tmp_mask_offset);
}

/* remove padding from embedding layer to transformer blocks */
template <typename T>
__global__ void remove_sequence_length_padding(const T *src, T *tgt,
                                               const int *tmp_mask_offset,
                                               int *mask_offset, const int n) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  mask_offset[bid] = tmp_mask_offset[bid];
  const int src_seq_id = bid + mask_offset[bid];
  const int tgt_seq_id = bid;

  for (int i = tid; i < n; i += blockDim.x) {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template <typename T>
void remove_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                   const int *tmp_mask_offset,
                                                   int *mask_offset,
                                                   const int m, const int n) {
  // src: [batch_size*max_seq_len, hidden_dim]
  // tgt: [valid_word_num, hidden_dim]
  remove_sequence_length_padding<<<m, 256>>>(src, tgt, tmp_mask_offset,
                                             mask_offset, n);
}

template void remove_sequence_length_padding_kernelLauncher(
    const float *src, float *tgt, const int *tmp_mask_offset, int *mask_offset,
    const int m, const int n);

template void remove_sequence_length_padding_kernelLauncher(
    const half *src, half *tgt, const int *tmp_mask_offset, int *mask_offset,
    const int m, const int n);

/* add padding from transformer blocks to final output*/

template <typename T>
__global__ void rebuild_sequence_length_padding(const T *src, T *tgt,
                                                const int *mask_offset,
                                                const int n) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < n; i += blockDim.x) {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template <typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T *src, T *tgt,
                                                    const int *mask_offset,
                                                    const int m, const int n) {
  // src: [valid_word_num, hidden_dim]
  // tgt: [batch_size*max_seq_len, hidden_dim]
  rebuild_sequence_length_padding<<<m, 256>>>(src, tgt, mask_offset, n);
}

template void
rebuild_sequence_length_padding_kernelLauncher(const float *src, float *tgt,
                                               const int *mask_offset,
                                               const int m, const int n);

template void
rebuild_sequence_length_padding_kernelLauncher(const half *src, half *tgt,
                                               const int *mask_offset,
                                               const int m, const int n);

/* FT transpose and remove padding */

template <typename T>
__global__ void
transpose_rebuild_padding(T *Q, T *K, T *V, T *q_buf_, T *k_buf_, T *v_buf_,
                          const int batch_size, const int seq_len,
                          const int head_num, const int size_per_head,
                          const int *mask_offset) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head +
                     tgt_head_id * seq_len * size_per_head +
                     tgt_seq_id * size_per_head + tgt_hidden_id;

  q_buf_[tgt_id] = Q[src_id]; // + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id]; // + bias_K[tid];
  v_buf_[tgt_id] = V[src_id]; // + bias_V[tid];
}

template <typename T>
void transpose_rebuild_padding_kernelLauncher(
    T *Q, T *K, T *V, T *q_buf, T *k_buf, T *v_buf, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int valid_word_num, const int *mask_offset) {
  const int k = head_num * size_per_head;

  if (std::is_same<T, float>::value) {
    transpose_rebuild_padding<<<valid_word_num, k>>>(
        Q, K, V, q_buf, k_buf, v_buf, batch_size, seq_len, head_num,
        size_per_head, mask_offset);
  } else {
    transpose_rebuild_padding<<<valid_word_num, k / 2>>>(
        (half2 *)Q, (half2 *)K, (half2 *)V, (half2 *)q_buf, (half2 *)k_buf,
        (half2 *)v_buf, batch_size, seq_len, head_num, size_per_head / 2,
        mask_offset);
  }
}

template void transpose_rebuild_padding_kernelLauncher(
    float *Q, float *K, float *V, float *q_buf, float *k_buf, float *v_buf,
    const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const int valid_word_num, const int *mask_offset);

template void transpose_rebuild_padding_kernelLauncher(
    half *Q, half *K, half *V, half *q_buf, half *k_buf, half *v_buf,
    const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const int valid_word_num, const int *mask_offset);

/* FT rebuild padding and transpose */
template <typename T>
__global__ void transpose_remove_padding(T *src, T *dst, const int batch_size,
                                         const int seq_len, const int head_num,
                                         const int size_per_head,
                                         const int *mask_offset) {
  // TODO: optimize this kernel?
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x;  // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] =
      src[src_batch_id * head_num * seq_len * size_per_head +
          head_id * seq_len * size_per_head + src_seq_id * size_per_head +
          hidden_id];
}

template <typename T>
void transpose_remove_padding_kernelLauncher(
    T *src, T *dst, const int valid_word_num, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int *mask_offset) {
  int k = head_num * size_per_head;
  if (std::is_same<T, float>::value) {
    transpose_remove_padding<<<valid_word_num, k>>>(
        src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset);
  } else {
    transpose_remove_padding<half2><<<valid_word_num, k / 2>>>(
        (half2 *)src, (half2 *)dst, batch_size, seq_len, head_num,
        size_per_head / 2, mask_offset);
  }
}

template void transpose_remove_padding_kernelLauncher(
    float *src, float *dst, const int valid_word_num, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int *mask_offset);

template void transpose_remove_padding_kernelLauncher(
    half *src, half *dst, const int valid_word_num, const int batch_size,
    const int seq_len, const int head_num, const int size_per_head,
    const int *mask_offset);