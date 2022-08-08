import time

from energonai.nemesis.nemesis_manager import Ne_manager
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

compute_device = 'cuda:0'  # manually set which device to compute on
offload_flag = True  # whether or not to activate offloading

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class single_linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias=False):
        super().__init__()
        self.weight = torch.empty(output_dim, input_dim)
        nn.init.normal_(self.weight)
        self.weight = nn.Parameter(self.weight.to(compute_device))
        if bias:
            self.bias = torch.empty(output_dim)
            nn.init.normal_(self.bias)
            self.bias = nn.Parameter(self.bias.to(compute_device))
        else:
            self.bias = None

    def forward(self, input_):
        output = F.linear(input_, self.weight, self.bias)
        return output


class nv_layers(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layer_num: int):
        super().__init__()
        self.module_list = list()
        for i in range(layer_num):
            if i == 0:
                temp_layer = single_linear(input_dim, hidden_dim, True)
            elif i == layer_num - 1:
                temp_layer = single_linear(hidden_dim, output_dim, True)
            else:
                temp_layer = single_linear(hidden_dim, hidden_dim, True)
            Ne_manager.register_module(temp_layer, compute_device)
            if Ne_manager.offload_flags[i] and offload_flag:
                Ne_manager.offload_module(temp_layer)
            self.module_list.append(temp_layer)

    def print_device(self):
        cnt__ = 0
        print("=" * 50)
        for mod in self.module_list:
            print("layer {} device: ".format(cnt__))
            cnt__ += 1
            print(next(mod.parameters()).data.device)
        print("=" * 50)

    def forward(self, input_):
        output = input_
        for layer_ in self.module_list:
            if Ne_manager.event_dict[id(layer_)] is not None:
                Ne_manager.compute_stream.wait_event(Ne_manager.event_dict[id(layer_)])
                Ne_manager.event_dict[id(layer_)] = None
            output = layer_(output)
        return output


if __name__ == "__main__":
    setup_seed(42)
    Ne_manager.set_model_info(12, 6)  # register model info
    Ne_manager.set_free_device("cuda:1")
    # Ne_manager.set_free_device("cpu")  # modify here if you want to use cpu as offloading target
    model_ = nv_layers(200, 150000, 10, 12)
    if offload_flag:
        Ne_manager.apply_hook()  # call this to activate offloading hooks
    input_ = torch.randn((20, 200)).to("cuda:0")
    print("init done")
    with torch.inference_mode():
        for i in range(5):
            out_ = model_(input_)
    start_ = time.time()
    with torch.inference_mode():
        for i in range(20):
            out_ = model_(input_)
    print(time.time() - start_)

