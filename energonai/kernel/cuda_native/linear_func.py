from typing_extensions import Self
import torch
import importlib

try:
    energonai_linear = importlib.import_module("energonai_linear_func")
except ImportError:
    raise RuntimeError('energonai_linear_func requires cuda extensions')


class MLPGemm(object):
    def __init__(self):
        self.start_algo = energonai_linear.get_start_algo()
        self.end_algo = energonai_linear.get_end_algo()
        self.start_algo_t_op = energonai_linear.get_start_algo_t_op()
        self.end_algo_t_op = energonai_linear.get_end_algo_t_op()
    
    def mlp_gemm(self, tensor1, tensor2, algo):
        return energonai_linear.mlp_gemm(tensor1, tensor2, algo)
