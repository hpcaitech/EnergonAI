from vit import vit_lite_depth7_patch4_32, vit_tiny_patch4_32, vit_base_patch16_224
from vit import vit_base_patch16_384, vit_base_patch32_224, vit_base_patch32_384, vit_large_patch16_224
from vit import vit_large_patch16_384, vit_large_patch32_224, vit_large_patch32_384
from vit_server import launch_engine


# for engine
model_class = vit_base_patch16_224
model_type = "vit"
host = "127.0.0.1"
port = 29402
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
