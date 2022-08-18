from hf_gpt2 import hf_gpt2
from hf_gpt2_server import launch_engine

# for engine
model_class = hf_gpt2
model_type = "gpt"
host = "127.0.0.1"
port = 29401
half = True
checkpoint = "/workspace/hf_gpt2/GPT2.bin"
backend = "nccl"

# for parallel 
tp_init_size = 2
pp_init_size = 2

# for server
engine_server = launch_engine
tokenizer_path = "/workspace/hf_gpt2"
server_host = "127.0.0.1"
server_port = 8020
log_level = "info"
