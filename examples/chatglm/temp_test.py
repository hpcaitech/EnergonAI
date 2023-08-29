from colossalai.nn import LayerNorm1D
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size=4096
        layernorm_epsilon=1e-5
        dtype=torch.float16
        self.input_layernorm = LayerNorm1D(hidden_size, eps=layernorm_epsilon, dtype=dtype)

    def forward(self,inputs):
        res=self.input_layernorm(inputs)
        print(self.input_layernorm)
        return res
    

data=torch.randn(4,1,4096)
data=data.clone().detach().to(device='cuda:3',dtype=torch.float16)
print(data.shape)
print(data[0][0][0])

diymodel=Model().to(device='cuda:3')
diymodel.eval()
res=diymodel(data)

print(res.shape)
print(res[0][0][0])