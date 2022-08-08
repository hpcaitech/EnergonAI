from energonai.kernel import linear, find_algo
import torch
import time


@torch.no_grad()
def test_linear_func():
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    inputs = torch.randn(batch_size, seq_len, din).half().cuda()
    params = torch.randn(dout, din).half().cuda()
    tensor_target = torch.nn.functional.linear(inputs, params)
    tensor_output = linear(inputs, params)
    diff = torch.abs(tensor_output - tensor_target)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    max_array = torch.max(tensor_target)

    if mean_diff > 0.5 or max_diff > 15 or max_diff / max_array > 0.05:
        print("mean_diff:%.2f, max_diff:%.2f, max_diff/max_array:%.4f" %
              (mean_diff, max_diff, max_diff / max_array))
        print('target:', tensor_target, '\n')
        print('output:', tensor_output, '\n')
        raise AssertionError("Wrong value!")

    print('tests pass')


@torch.no_grad()
def benchmark_linear_func():
    algo = find_algo()
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    inner_loop = 8
    outer_loop = 20

    input_list_1 = []
    param_list_1 = []
    input_list_2 = []
    param_list_2 = []
    for i in range(inner_loop):
        input_list_1.append(torch.randn(batch_size, seq_len, din).half().cuda())
        param_list_1.append(torch.randn(dout, din).half().cuda())
        input_list_2.append(input_list_1[-1].clone().detach())
        param_list_2.append(param_list_1[-1].clone().detach())

    torch_count = 0        
    cublas_count = 0

    for _ in range(outer_loop):
        for i in range(inner_loop):
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            _ = linear(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            _ = linear(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            start_time = time.time()
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            torch_count += time.time() - start_time
            
            torch.cuda.synchronize()
            start_time = time.time()
            _ = linear(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()
            cublas_count += time.time() - start_time
    
    torch_time = torch_count / inner_loop / outer_loop
    cublas_time = (cublas_count / inner_loop / outer_loop)
    print("==>  torch time: %.6f" % torch_time)
    print("==> cublas time: %.6f, speedup: %.4f%%" % (cublas_time, (torch_time - cublas_time) / torch_time * 100))


if __name__ == '__main__':
    test_linear_func()
    benchmark_linear_func()
