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
fixed_cache_keys = [
    ('Question: What is the name of the largest continent on earth?\nAnswer: Asia\n\nQuestion: What is at the center of the solar system?\nAnswer:', 64),
    ('A chat between a salesman and a student.\n\nSalesman: Hi boy, are you looking for a new phone?\nStudent: Yes, my phone is not functioning well.\nSalesman: What is your budget? \nStudent: I have received my scholarship so I am fine with any phone.\nSalesman: Great, then perhaps this latest flagship phone is just right for you.', 64),
    ("English: I am happy today.\nChinese: 我今天很开心。\n\nEnglish: I am going to play basketball.\nChinese: 我一会去打篮球。\n\nEnglish: Let's celebrate our anniversary.\nChinese:", 64)
]
