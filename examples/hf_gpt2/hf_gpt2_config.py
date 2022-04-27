from hf_gpt2 import hf_gpt2
from hf_gpt2_server import launch_engine

model_class = hf_gpt2
model_type = "gpt"
engine_server = launch_engine
tp_init_size = 2
pp_init_size = 2
host = "127.0.0.1"
port = 29400
half = True
checkpoint = "/home/lcdjs/hf_gpt2/GPT2.bin"
tokenizer_path = "/home/lcdjs/hf_gpt2"
server_host = "127.0.0.1"
server_port = 8010
log_level = "info"
backend = "nccl"