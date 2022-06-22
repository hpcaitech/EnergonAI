import sys
import torch
import pynvml

NUM_EXPAND = 1024 ** 3


class gpu_info:

    def __init__(self, device_id: int):
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.mem_used = info.used / NUM_EXPAND
        self.mem_free = info.free / NUM_EXPAND
        self.mem_managed = 0
        self._hold_module_list = list()

    def update_mem_state(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.mem_used = info.used / NUM_EXPAND
        self.mem_free = info.free / NUM_EXPAND

    def gpu_register_module(self, module_: torch.nn.Module):
        self._hold_module_list.append(module_)
        self.update_mem_state()

    def release_module(self, module_: torch.nn.Module):
        self._hold_module_list = [t for t in self._hold_module_list if id(t) != id(module_)]

    def check_avail_mem(self, required_space: float):
        self.update_mem_state()
        if self.mem_free > required_space:
            return True
        else:
            return False

    def print_status(self):
        print("mem used: {} mem free: {} mem managed: {},"
              " \n".format(self.mem_used,
                           self.mem_free,
                           self.mem_managed,

                           ))


class Nemesis_Manager:

    def __init__(self):
        pynvml.nvmlInit()
        self._gpu_num = pynvml.nvmlDeviceGetCount()
        self._gpu_info = {"cuda:{}".format(i): gpu_info(i) for i in range(self._gpu_num)}
        self._module_list = list()
        self.offload_dict = dict()
        self.offload_flags = None
        self.prefetch_dict = dict()
        self.compute_device_dict = dict()
        self.module_size = dict()
        self.layer_num = -1
        self.offload_interval = -1
        self.free_device = None
        self._model = None

    def register_model(self, model_):
        self._model = model_

    def set_free_device(self, free_device):
        self.free_device = free_device

    def set_model_info(self, layer_num, offload_interval):
        assert layer_num % offload_interval == 0
        self.layer_num = layer_num
        self.offload_interval = offload_interval
        self.generate_offload_dict()

    def calculate_module_size(self, module_: torch.nn.Module):
        res_size = 0
        for ts in module_.parameters():
            res_size += ts.data.numel() * ts.data.element_size()
        res_size /= NUM_EXPAND
        return res_size

    def move_module(self, module_: torch.nn.Module, target_device):
        module_ = module_.to(target_device, non_blocking=True)

    def generate_offload_dict(self):
        assert self.layer_num != -1 and self.offload_interval != -1, 'please set layer num and offload interval first'
        res_dict = {i: False for i in range(self.layer_num)}
        for j in range(self.offload_interval - 1, self.layer_num, self.offload_interval):
            res_dict[j] = True
        self.offload_flags = res_dict

    def offload_module(self, module_):
        free_device = self.free_device
        if free_device is None:
            raise AssertionError("please call set_free_device function first")
        Ne_manager.move_module(module_, free_device)

    def apply_hook(self):
        for i in range(len(self._module_list)):
            if i % self.offload_interval == 0:
                self.offload_dict[id(self._module_list[i])].append(self._module_list[i - 1])
            if i % self.offload_interval == self.offload_interval - 3:
                self.prefetch_dict[id(self._module_list[i])].append(self._module_list[i + 2])
            if len(self.prefetch_dict[id(self._module_list[i])]) + len(self.offload_dict[id(self._module_list[i])]) > 0:
                self._module_list[i].register_forward_pre_hook(basic_hook)

    def register_module(self, module_: torch.nn.Module, device: str):
        self._gpu_info[device].gpu_register_module(module_)
        self.offload_dict[id(module_)] = list()
        self.prefetch_dict[id(module_)] = list()
        self._module_list.append(module_)
        self.compute_device_dict[id(module_)] = device
        self.module_size[id(module_)] = self.calculate_module_size(module_)

    def find_free_gpu(self, size: float, ori_gpu):
        if isinstance(ori_gpu, torch.device):
            ori_gpu = "{}:{}".format(ori_gpu.type, ori_gpu.index)
        for gpu_name, gpu_ in self._gpu_info.items():
            if gpu_name == ori_gpu:
                continue
            if gpu_.check_avail_mem(size):
                return gpu_name
        print("Error: No available gpu memory")
        sys.exit()

    def print_status(self):
        print("=" * 60)
        for gpu_info_ in self._gpu_info.values():
            gpu_info_.print_status()
        print("=" * 60)


Ne_manager = Nemesis_Manager()


def basic_hook(module: torch.nn.Module, input_):
    for tg in Ne_manager.offload_dict[id(module)]:
        cur_device = next(tg.parameters()).device
        if Ne_manager.compute_device_dict[id(tg)] == "{}:{}".format(cur_device.type, cur_device.index):
            free_device = Ne_manager.free_device
            if free_device is None:
                raise AssertionError("please call set_free_device function first")
            Ne_manager.move_module(tg, free_device)
    for tg in Ne_manager.prefetch_dict[id(module)]:
        Ne_manager.move_module(tg, Ne_manager.compute_device_dict[id(tg)])
    return
