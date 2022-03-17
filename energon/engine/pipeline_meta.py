import torch
from energon.core import global_context as gpc
from energon.context import ParallelMode

class PipelineMeta:
    def __init__(self, tensor_num_dim: int = -1, max_batch_size: int = -1):
        # Tell pipeline the information of the next batch
        self.tensor_num_dim = tensor_num_dim
        self.max_batch_size = max_batch_size

        self.info_len = tensor_num_dim + max_batch_size + 1
        self.batch_size = 1 # 1
        
        self.tensor_shapes = [] # ([32, 512, 1600])  3
        self.seq_lens = [] # [42,52,63,12] # batch_size
        self.meta_tensor = torch.zeros(self.info_len, dtype = torch.int, requires_grad=False).cuda()
 

    # resolve meta from tensor.
    # batchsize | TensorShape | seq_lens
    def store_meta(self, metaTensor):
        self.meta_tensor = metaTensor

        cur = 0
        self.batch_size = metaTensor[0].item()
        # print(type(self.batch_size))
        
        self.tensor_shapes.clear()
        for i in range(self.tensor_num_dim):
            cur = cur + 1
            self.tensor_shapes.append(metaTensor[cur].item())

        self.seq_lens.clear()
        for i in range(self.batch_size):
            cur = cur + 1
            self.seq_lens.append(metaTensor[cur].item())


    def get_tensor_num_dim(self):
        return self.tensor_num_dim

    def get_tensor_shapes(self):
        return torch.Size(self.tensor_shapes)
    
    def get_batch_size(self):
        return self.batch_size

    def get_seq_lens(self):
        return self.seq_lens

    def get_meta_tensor(self):
        # print(self.meta_tensor)
        return self.meta_tensor

    def get_meta_tensor_shape(self):
        return self.meta_tensor.size()


    def get_info_len(self):
        return self.info_len

    
