/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/th_op/decoder.h"

namespace torch_ext {
using torch::Tensor;

FasterTransformerDecoder::FasterTransformerDecoder(
  int64_t head_num,
  int64_t head_size,
  int64_t mem_hidden_dim,
  Tensor self_layernorm_gamma,
  Tensor self_layernorm_beta,
  Tensor self_kernel_q,
  Tensor self_kernel_k,
  Tensor self_kernel_v,
  Tensor self_bias_q,
  Tensor self_bias_k,
  Tensor self_bias_v,
  Tensor self_output_kernel,
  Tensor self_output_bias,
  Tensor cross_layernorm_gamma,
  Tensor cross_layernorm_beta,
  Tensor cross_kernel_q,
  Tensor cross_kernel_k,
  Tensor cross_kernel_v,
  Tensor cross_bias_q,
  Tensor cross_bias_k,
  Tensor cross_bias_v,
  Tensor cross_output_kernel,
  Tensor cross_output_bias,
  Tensor ffn_layernorm_gamma,
  Tensor ffn_layernorm_beta,
  Tensor inter_kernel,
  Tensor inter_bias,
  Tensor output_kernel,
  Tensor output_bias,
  c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg)
: _st(self_layernorm_gamma.scalar_type()),
  weights{self_layernorm_gamma, self_layernorm_beta,
          self_kernel_q, self_kernel_k, self_kernel_v, self_bias_q, self_bias_k, self_bias_v, self_output_kernel, self_output_bias,
          cross_layernorm_gamma, cross_layernorm_beta,
          cross_kernel_q, cross_kernel_k, cross_kernel_v, cross_bias_q, cross_bias_k, cross_bias_v,
          cross_output_kernel, cross_output_bias,
          ffn_layernorm_gamma, ffn_layernorm_beta, inter_kernel, inter_bias, output_kernel, output_bias}
{
  CHECK_INPUT(self_layernorm_gamma, _st);  // hidden_dim
  CHECK_INPUT(self_layernorm_beta, _st);  // hidden_dim
  CHECK_INPUT(self_kernel_q, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_kernel_k, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_kernel_v, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_bias_q, _st);  // hidden_dim
  CHECK_INPUT(self_bias_k, _st);  // hidden_dim
  CHECK_INPUT(self_bias_v, _st);  // hidden_dim
  CHECK_INPUT(self_output_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_output_bias, _st);  // hidden_dim
  CHECK_INPUT(cross_layernorm_gamma, _st);  // hidden_dim
  CHECK_INPUT(cross_layernorm_beta, _st);  // hidden_dim
  CHECK_INPUT(cross_kernel_q, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(cross_kernel_k, _st);  // mem_hidden_dim, hidden_dim
  CHECK_INPUT(cross_kernel_v, _st);  // mem_hidden_dim, hidden_dim
  CHECK_INPUT(cross_bias_q, _st);  // hidden_dim
  CHECK_INPUT(cross_bias_k, _st);  // hidden_dim
  CHECK_INPUT(cross_bias_v, _st);  // hidden_dim
  CHECK_INPUT(cross_output_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(cross_output_bias, _st);  // hidden_dim
  CHECK_INPUT(ffn_layernorm_gamma, _st);  // hidden_dim
  CHECK_INPUT(ffn_layernorm_beta, _st);  // hidden_dim
  CHECK_INPUT(inter_kernel, _st);  // hidden_dim, 4 * hidden_dim
  CHECK_INPUT(inter_bias, _st);  // 4 * hidden_dim
  CHECK_INPUT(output_kernel, _st);  // 4 * hidden_dim, hidden_dim
  CHECK_INPUT(output_bias, _st);  // hidden_dim

  pg_ = pg;
  // convert ncclpg into parallel_param
  int rank = pg_->getRank();
  int world_size = pg_->getSize();;
  int local_head_num = head_num / world_size;
  int local_hidden_units = local_head_num * head_size;

  // init commnicator
  ncclUniqueId tensor_para_nccl_uid;
  ncclComm_t tensor_para_nccl_comm;
  NCCLCHECK( ncclGetUniqueId(&tensor_para_nccl_uid));
  NCCLCHECK( ncclCommInitRank(&tensor_para_nccl_comm, world_size, tensor_para_nccl_uid, rank));



  t_parallel_param_.rank = rank;
  t_parallel_param_.world_size = world_size;
  t_parallel_param_.nccl_comm = tensor_para_nccl_comm;
  t_parallel_param_.local_head_num_ = local_head_num;
  t_parallel_param_.local_hidden_units_ = local_hidden_units;


  switch (_st) {
    case at::ScalarType::Float:
      ftdecoder = new torch_ext::FTDecoder<float>(head_num, head_size, mem_hidden_dim, weights, t_parallel_param_);
      break;
    case at::ScalarType::Half:
      ftdecoder = new torch_ext::FTDecoder<half>(head_num, head_size, mem_hidden_dim, weights, t_parallel_param_);
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
  head_info = torch::empty({3}, torch::dtype(torch::kInt64));
  head_info[0] = head_num;
  head_info[1] = head_size;
  head_info[2] = mem_hidden_dim;
}

FasterTransformerDecoder::~FasterTransformerDecoder() {
  delete ftdecoder;
}
  
Tensor FasterTransformerDecoder::forward(Tensor input, Tensor memory, Tensor memory_seq_lens, std::vector<Tensor> self_cache, Tensor mem_cache, int64_t step) {
  CHECK_INPUT(input, _st);
  CHECK_INPUT(memory, _st);
  CHECK_INPUT(self_cache[0], _st);
  CHECK_INPUT(self_cache[1], _st);
  CHECK_INPUT(mem_cache, _st);
  CHECK_CUDA(memory_seq_lens); CHECK_CONTIGUOUS(memory_seq_lens); TORCH_CHECK(memory_seq_lens.dtype()==torch::kInt32, "mem_seq_lens dtype should be int32");
  auto mem_size = memory.sizes();
  int batch_size = mem_size[0];
  int seq_len = mem_size[1];
  int mem_hidden_dim = mem_size[2];
  auto output = torch::empty_like(input);
  ftdecoder->forward(batch_size, seq_len, mem_hidden_dim, step, input, memory, memory_seq_lens, self_cache, mem_cache, output);
  return output;
}

std::vector<Tensor> FasterTransformerDecoder::get_pickle_info() const {
  std::vector<Tensor> tmp(weights);
  tmp.push_back(head_info);
  return tmp;
}
} // namespace torch_ext