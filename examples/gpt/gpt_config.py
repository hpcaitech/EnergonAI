from gpt import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from gpt_batch_server import launch_engine

# for engine
model_class = gpt2_large
model_type = "gpt"
host = "127.0.0.1"
port = 29401
half = True
backend = "nccl"

# for parallel
tp_init_size = 2
pp_init_size = 2

# for server
engine_server = launch_engine
server_host = "127.0.0.1"
server_port = 8020
log_level = "info"
tokenizer_path = "/home/lcdjs/hf_gpt2"
rm_padding = False
