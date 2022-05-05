from gpt import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from gpt_batch_server import launch_engine

model_class = gpt2_large
model_type = "gpt"
engine_server = launch_engine
tp_init_size = 2
pp_init_size = 2
host = "127.0.0.1"
port = 29400
half = True
server_host = "127.0.0.1"
server_port = 8020
log_level = "info"
backend = "nccl"
tokenizer_path = "/home/lcdjs/hf_gpt2"