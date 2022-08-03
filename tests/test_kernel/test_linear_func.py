from energonai.kernel import EnergonLinearFunc
import torch
import time
import pytest


batch_size = 64
seq_len = 2048
din = 12288
dout = 12288


def test_linear_func():
    linear = EnergonLinearFunc()

    tensor1 = torch.rand(batch_size, seq_len, din).half().cuda()
    tensor2 = torch.rand(din, dout).half().cuda()

    warmup = 5
    loop = 5
    for i in range(warmup):
        linear.mlp_gemm(tensor1, tensor2)
        tensor_target = torch.matmul(tensor1, tensor2)

    start_time = time.time()
    for i in range(loop):
        tensor_target = torch.matmul(tensor1, tensor2)
    print("==> torch time: %.6f" % ((time.time() - start_time) / loop))

    start_time = time.time()
    for algo in range(linear.get_start_algo(), linear.get_end_algo() + 1):
        for i in range(loop):
            tensor_output = linear.mlp_gemm(tensor1, tensor2, algo)
        print("==> cublas time: %.6f algo:%d" % ((time.time() - start_time) / loop, algo))

    print('target:', tensor_target, '\n')
    print('output:', tensor_output, '\n')
    # assert torch.equal(tensor_target, tensor_output)

if __name__ == '__main__':
    test_linear_func()