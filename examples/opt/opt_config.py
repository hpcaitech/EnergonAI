from energonai.model import opt_30B, opt_125M
from opt_server import launch_engine

# for engine
model_class = opt_125M
model_type = "gpt"
host = "127.0.0.1"
port = 29402
half = True
checkpoint = "/data/user/djs_model_checkpoint/opt_metaseq_125m/model/restored.pt"
#"/data/user/djs_model_checkpoint/opt-30B-singleton/opt_metaseq_30000m/model/restored.pt"
backend = "nccl"

# for parallel 
tp_init_size = 2
pp_init_size = 2

# for server
engine_server = launch_engine
tokenizer_path = "facebook/opt-350m"
# server_host = "127.0.0.1"
server_host = "0.0.0.0"
server_port = 8020
log_level = "info"