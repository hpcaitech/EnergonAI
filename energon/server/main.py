from worker_server import launch_worker
from engine_server import launch_engine

launch_engine("bert_small", "bert", 32, 1, 1)
