#include <c10/util/intrusive_ptr.h>
#include <c10d/NCCLUtils.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/Types.hpp>

#include "nccl.h"
#include <iostream>
#include <string>
#include <torch/extension.h>

// c10::intrusive_ptr<c10d::ProcessGroup::Work>
void sendNcclUniqueId(at::Tensor &ncclid, int dstRank,
                      const c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg) {
  // pack in
  // auto tensor = torch::from_blob(ncclId->internal, {int(32)},
  // torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32).requires_grad(false));
  // at::Tensor tensor = torch::zeros({int(32)},
  // torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32));
  std::vector<at::Tensor> tensors = {ncclid};
  printf("[INFO] rank start send \n");

  if (pg == c10::detail::UniqueVoidPtr()) {
    auto ret = pg->send(tensors, dstRank, 0);
    ret->wait();
  }

  printf("[INFO] rank finish send \n");
  // return ret;
}

void recvNcclUniqueId(at::Tensor &ncclid, int srcRank,
                      const c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg) {
  // pack in
  at::Tensor tensor = torch::zeros(
      {int(32)}, torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32));
  // auto tensor = torch::from_blob(ncclId->internal, {int(32)},
  // torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32).requires_grad(false));
  std::vector<at::Tensor> tensors = {ncclid};
  printf("[INFO] rank start recv \n");

  if (pg == c10::detail::UniqueVoidPtr()) {
    auto ret = pg->recv(tensors, srcRank, 0);
    ret->wait();
  }

  printf("[INFO] rank finish recv \n");
  // at::Tensor tensor = tensors[0];
  // float* temp = tensor.data_ptr<float>();
  // ncclId->internal
  // char * x = reinterpret_cast<char*>(temp);
  // get_ptr<ncclUniqueId>(tensor);
}
//   if(local_rank == 0)
//   {
//     for(int i = 1; i<tensor_para_size; i++){
//       printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n",
//       int(rank), int(rank + i)); sendNcclUniqueId(&tensor_para_nccl_uid,
//       rank+i, pg);
//     }
//   }else{
//       printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n",
//       int(rank), int(rank - local_rank));
//       recvNcclUniqueId(&tensor_para_nccl_uid ,rank - local_rank, pg);
//   }
// std::string res(tensor_para_nccl_uid.internal, NCCL_UNIQUE_ID_BYTES);

// ncclUniqueId* ncclId, int srcRank,
void broadcastUniqueId(at::Tensor &ncclid, int local_rank,
                       const c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg) {

  std::vector<at::Tensor> tensors = {ncclid};

  printf("[INFO] rank start ncclid broadcast \n");

  if (pg != c10::detail::UniqueVoidPtr()) {
    auto ret = pg->broadcast(tensors, c10d::BroadcastOptions());
    ret->wait();
  }

  printf("[INFO] rank finish ncclid broadcast in func \n");

  // char* temp = reinterpret_cast<char*>(cpuNCCLID.data_ptr<float>());
  // for(int i = 0; i<NCCL_UNIQUE_ID_BYTES; i++){
  //   std::cout<<temp[i]-48<<",";
  // }
}

// if(local_rank == 0)
//     {
//       for(int i = 1; i<tensor_para_size; i++){
//         printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n",
//         int(rank), int(rank + i)); sendNcclUniqueId(tensor, rank+i, pg);
//       }
//     }else{
//         printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d
//         \n", int(rank), int(rank - local_rank)); recvNcclUniqueId(tensor,rank
//         - local_rank, pg);
//     }

// #define NCCL_UNIQUE_ID_BYTES 128
// typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
// std::string
at::Tensor getNCCLInitID(int64_t tensor_para_size, int64_t local_rank,
                         const c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg) {

  ncclUniqueId tensor_para_nccl_uid;
  ncclGetUniqueId(&tensor_para_nccl_uid);
  auto tensor = torch::from_blob(tensor_para_nccl_uid.internal, {int(32)},
                                 torch::TensorOptions(torch::kCPU)
                                     .dtype(torch::kFloat32)
                                     .requires_grad(false));
  torch::Tensor gpuNCCLID = tensor.to(torch::kCUDA);
  broadcastUniqueId(gpuNCCLID, local_rank, pg);
  torch::Tensor cpuNCCLID = gpuNCCLID.to(torch::kCPU);

  // char* temp = reinterpret_cast<char*>(cpuNCCLID.data_ptr<float>());
  // for(int i = 0; i<NCCL_UNIQUE_ID_BYTES; i++){
  //   std::cout<<temp[i]-48<<",";
  // }

  return cpuNCCLID;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_nccl", &getNCCLInitID, "GET NCCL UNIQUE ID");
}

// printf("[INFO] rank %d get ncclid \n", int(local_rank));
// for(int i = 0; i<NCCL_UNIQUE_ID_BYTES; i++){
//   std::cout<<tensor_para_nccl_uid.internal[i]-48<<",";
// }
// float* temp = new float(32);

// at::Tensor tensor = torch::zeros({int(32)},
// torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32));

// for(int i = 0; i<NCCL_UNIQUE_ID_BYTES; i++){
//   std::cout<<temp[i]-48<<",";
// }
