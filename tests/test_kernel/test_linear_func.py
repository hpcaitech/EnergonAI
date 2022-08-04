from energonai.kernel import EnergonLinearFunc
import torch
import time


@torch.no_grad()
def test_linear_func():
    batch_size = 2
    seq_len = 3
    din = 4
    dout = 5
    
    energon_linear = EnergonLinearFunc()

    inputs = torch.randn(batch_size, seq_len, din).half().cuda()
    params = torch.randn(din, dout).half().cuda()
    tensor_target = torch.matmul(inputs, params)
    tensor_output = energon_linear.mlp_gemm(inputs, params, energon_linear.get_start_algo())
    
    diff = torch.abs(tensor_output - tensor_target)
    diff = torch.mean(diff) / din / dout
    if diff > 5e-5:
        print(torch.mean(torch.abs(tensor_output - tensor_target)))
        print('target:', tensor_target, '\n')
        print('output:', tensor_output, '\n')
        raise AssertionError("Wrong value!")

    print('Tests pass!')


@torch.no_grad()
def benchmark_linear_func():
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152
    
    loop = 15
    energon_linear = EnergonLinearFunc()
    
    input_list_1 = []
    param_list_1 = []
    input_list_2 = []
    param_list_2 = []
    for i in range(loop):
        input_list_1.append(torch.randn(batch_size, seq_len, din).half().cuda())
        param_list_1.append(torch.randn(din, dout).half().cuda())
        input_list_2.append(input_list_1[-1].clone().detach())
        param_list_2.append(param_list_1[-1].clone().detach().T)

    energon_count = 0
    torch_count = 0
    for i in range(loop):
        _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
        _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], 17)
        _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
        _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], 17)
        
        start_time = time.time()
        _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
        torch_count += time.time() - start_time
        
        start_time = time.time()
        _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], 17)
        energon_count += time.time() - start_time
    
    print("==> torch time: %.6f" % (torch_count / loop))
    print("==> cublas time: %.6f algo:%d" % (energon_count / loop, 17))

    
if __name__ == '__main__':
    test_linear_func()
    benchmark_linear_func()
