from bert import bert_small, bert_large, bert_xl, bert_8B, bert_175B
from bert_server import launch_engine

model_class = bert_8B
model_type = "bert"
engine_server = launch_engine
tp_init_size = 2
pp_init_size = 2
host = "127.0.0.1"
port = 29400
half = False
server_host = "127.0.0.1"
server_port = 8010
log_level = "info"
backend = "nccl"