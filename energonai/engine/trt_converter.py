from torch2trt.torch2trt import *
                
@tensorrt_converter('torch.matmul')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, trt.MatrixOperation.NONE, input_b_trt, trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)