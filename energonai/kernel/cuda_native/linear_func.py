import torch
import importlib

try:
    energonai_linear = importlib.import_module("energonai_linear_func")
except ImportError:
    raise RuntimeError('energonai_linear_func requires cuda extensions')


def linear(inputs, param, algo=energonai_linear.get_start_algo()):
    """
    linear function using Cublas
    
    Args:
        inputs (tensor): (batch, seq_len, din)
        param (tensor): (dout, din)
        algo (int): Cublas GEMM algorithms, defaults to CUBLAS_GEMM_DEFAULT. No effect for Ampere architecture gpu or above.
    Returns:
        tensor: (batch, seq_len, dout)
    """
    assert inputs.is_contiguous()
    assert param.is_contiguous()
    assert len(inputs.shape) == 3
    assert len(param.shape) == 2
    assert inputs.shape[2] == param.shape[1]
    return energonai_linear.mlp_gemm(inputs, param, algo)
