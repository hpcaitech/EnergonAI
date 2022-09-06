from energonai.model import opt_30B, opt_125M, opt_175B
from opt_server import launch_engine

# for engine
model_class = opt_175B
model_type = "gpt"
host = "127.0.0.1"
port = 29402
half = True
# checkpoint = "/data/user/djs_model_checkpoint/opt_metaseq_125m/model/restored.pt"

# If serving using a docker, map your own checkpoint directory to /model_checkpoint
checkpoint = '/data/user/lclhx/opt-175B/'
# "/data/user/djs_model_checkpoint/opt-30B-singleton/opt_metaseq_30000m/model/restored.pt"
backend = "nccl"

# for parallel
tp_init_size = 8
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
timeout_keep_alive = 180
executor_max_queue_size = 0
