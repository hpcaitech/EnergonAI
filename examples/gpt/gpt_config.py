from gpt import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from gpt_batch_server import launch_engine

# for engine
model_class = gpt2_8B
model_type = "gpt"
host = "127.0.0.1"
port = 29401
half = True
backend = "nccl"

# for parallel
tp_init_size = 4
pp_init_size = 2

# for server
engine_server = launch_engine
server_host = "127.0.0.1"
server_port = 8016
log_level = "info"
tokenizer_path = "/workspace/hf_gpt2"
rm_padding = False

#for batch manager
max_batch_size = 15
max_sequence_length = 1024
repeat_round = 2
step = 8
max_wait_time = 2