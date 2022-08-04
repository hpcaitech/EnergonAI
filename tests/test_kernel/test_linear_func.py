from energonai.kernel import EnergonLinearFunc
import torch
import time
import tqdm


@torch.no_grad()
def test_linear_func():
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    energon_linear = EnergonLinearFunc()

    inputs = torch.randn(batch_size, seq_len, din).half().cuda()
    params = torch.randn(din, dout).half().cuda()
    tensor_target = torch.nn.functional.linear(inputs, params.T)
    tensor_output = energon_linear.mlp_gemm(
        inputs, params, energon_linear.get_start_algo())

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
def find_algo():
    print("searching algo... ", end='')
    
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    inner_loop = 10
    energon_linear = EnergonLinearFunc()

    input_list = []
    param_list = []
    for i in range(inner_loop):
        input_list.append(torch.randn(batch_size, seq_len, din).half().cuda())
        param_list.append(torch.randn(din, dout).half().cuda())

    algo_map = {}
    for algo in range(energon_linear.get_start_algo(), energon_linear.get_end_algo() + 1):
        algo_map[algo] = 0
    for algo in range(energon_linear.get_start_algo_t_op(), energon_linear.get_end_algo_t_op() + 1):
        algo_map[algo] = 0

    for i in range(inner_loop):
        _ = energon_linear.mlp_gemm(input_list[i], param_list[i], energon_linear.get_start_algo())

        for algo in range(energon_linear.get_start_algo(), energon_linear.get_end_algo() + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = energon_linear.mlp_gemm(input_list[i], param_list[i], algo)
            torch.cuda.synchronize()
            algo_map[algo] += time.time() - start_time

        for algo in range(energon_linear.get_start_algo_t_op(), energon_linear.get_end_algo_t_op() + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = energon_linear.mlp_gemm(input_list[i], param_list[i], algo)
            torch.cuda.synchronize()
            algo_map[algo] += time.time() - start_time

    # print(algo_map)
    best_idx = None
    best_value = 999
    for key, value in algo_map.items():
        if value < best_value:
            best_value = value
            best_idx = key
    print("==> best algo: %d" % best_idx)
    return best_idx


@torch.no_grad()
def benchmark_linear_func():
    algo = find_algo()
    batch_size = 16
    seq_len = 64
    din = 12288
    dout = 49152

    inner_loop = 8
    outer_loop = 20
    energon_linear = EnergonLinearFunc()

    input_list_1 = []
    param_list_1 = []
    input_list_2 = []
    param_list_2 = []
    for i in range(inner_loop):
        input_list_1.append(torch.randn(batch_size, seq_len, din).half().cuda())
        param_list_1.append(torch.randn(din, dout).half().cuda())
        input_list_2.append(input_list_1[-1].clone().detach())
        param_list_2.append(param_list_1[-1].clone().detach().T)

    energon_count = 0
    torch_count = 0        

    for _ in range(outer_loop):
        for i in range(inner_loop):
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            start_time = time.time()
            _ = torch.nn.functional.linear(input_list_2[i], param_list_2[i])
            torch.cuda.synchronize()
            torch_count += time.time() - start_time
            
            torch.cuda.synchronize()
            start_time = time.time()
            _ = energon_linear.mlp_gemm(input_list_1[i], param_list_1[i], algo)
            torch.cuda.synchronize()
            energon_count += time.time() - start_time
            
    print("==> torch time: %.6f" % (torch_count / inner_loop / outer_loop))
    print("==> cublas time: %.6f " % (energon_count / inner_loop / outer_loop))


if __name__ == '__main__':
    test_linear_func()
    benchmark_linear_func()
