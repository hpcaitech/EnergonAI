import torch
import time
import importlib

try:
    energonai_linear = importlib.import_module("energonai_linear_func")
except ImportError:
    raise RuntimeError('energonai_linear_func requires cuda extensions')


def linear(inputs, param, algo=-1):
    """
    Linear function using Cublas

    Args:
        inputs (tensor): (batch, seq_len, din)
        param (tensor): (dout, din)
        algo (int): Cublas GEMM algorithms, defaults to -1. No effect for Ampere architecture gpu or above.
            -1: Apply Heuristics to select the GEMM algorithm
            0~23: Explicitly choose a GEMM algorithm
            99: Apply Heuristics to select the GEMM algorithm while allowing the use of Tensor Core operations if possible
            100~115: Explicitly choose a GEMM algorithm allowing it to use Tensor Core operations if possible
    Returns:
        tensor: (batch, seq_len, dout)
    """
    assert inputs.is_contiguous()
    assert param.is_contiguous()
    assert len(inputs.shape) == 3
    assert len(param.shape) == 2
    assert inputs.shape[2] == param.shape[1]
    assert isinstance(algo, int) and (-1 <= algo <= 23 or 99 <= algo <= 115)
    return energonai_linear.mlp_gemm(inputs, param, algo)


@torch.no_grad()
def find_algo():
    """
    Auto find best algo, may take tens of seconds
    
    Returns:
        int: best algo
    """
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    inner_loop = 3
    
    input_list = []
    param_list = []
    for i in range(inner_loop):
        input_list.append(torch.randn(batch_size, seq_len, din).half().cuda())
        param_list.append(torch.randn(dout, din).half().cuda())

    start_algo = -1
    end_algo = 23
    start_algo_t_op = 99
    end_algo_t_op = 115
    
    algo_map = {}
    for algo in range(start_algo, end_algo + 1):
        algo_map[algo] = 0
    for algo in range(start_algo_t_op, end_algo_t_op + 1):
        algo_map[algo] = 0

    for i in range(inner_loop):
        _ = linear(input_list[i], param_list[i], start_algo)
        _ = linear(input_list[i], param_list[i], start_algo)

        for algo in range(start_algo, end_algo + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = linear(input_list[i], param_list[i], algo)
            torch.cuda.synchronize()
            algo_map[algo] += time.time() - start_time

        for algo in range(start_algo_t_op, end_algo_t_op + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = linear(input_list[i], param_list[i], algo)
            torch.cuda.synchronize()
            algo_map[algo] += time.time() - start_time

    best_idx = None
    best_value = 999
    for key, value in algo_map.items():
        if value < best_value:
            best_value = value
            best_idx = key
        
    return best_idx
