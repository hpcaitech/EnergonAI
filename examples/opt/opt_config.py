from energonai.model import opt_30B, opt_125M, opt_66B
from opt_server import launch_engine

# for engine
model_class = opt_30B
model_type = "gpt"
host = "127.0.0.1"
port = 29402
half = True

# If serving using a docker, map your own checkpoint directory to /model_checkpoint
# checkpoint = '/model_checkpoint/'

# checkpoint = "/data/user/djs_model_checkpoint/opt_metaseq_125m/model/restored.pt"
checkpoint = "/data/user/lclhx/opt-30B"
# checkpoint="/data/user/djs_model_checkpoint/opt-66B-fragment"

backend = "nccl"

# for parallel
tp_init_size = 4
pp_init_size = 1

# for server
engine_server = launch_engine
# tokenizer_path = "facebook/opt-125m"
tokenizer_path = "facebook/opt-30b"
# tokenizer_path = "facebook/opt-66b"
server_host = "0.0.0.0"
server_port = 8020
log_level = "info"
allow_cors = True
executor_max_batch_size = 16
cache_size = 50
cache_list_size = 2
