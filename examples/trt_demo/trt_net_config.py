import torch
from net import bert_large
from trt_net_server import launch_engine


# for engine
model_class = bert_large
model_type = "bert"
host = "127.0.0.1"
port = 29401
half = True
backend = "nccl"

# for parallel 
tp_init_size = 1
pp_init_size = 1

# for server
engine_server = launch_engine
server_host = "127.0.0.1"
server_port = 8020
log_level = "info"

# for tensorrt
trt_sample = [torch.ones((1,128,1024)).half().cuda(), torch.ones((1, 1, 128)).half().cuda()]
