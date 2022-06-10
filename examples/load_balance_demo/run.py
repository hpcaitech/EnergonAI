from gpt import gpt2_8B_lb
# from bert import BertEmbedding1D, BertTransformerLayer1D
from colossalai import launch_from_torch

config = dict(parallel=dict(pipeline=dict(size=4), tensor=dict(size=1, mode='1d')))
launch_from_torch(config)


from energonai.context import mcfg
# mcfg.load_config("/home/lcdjs/EnergonAI/examples/auto_pipeline/bert_config.py")
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

models = gpt2_8B_lb()

print(f'Rank: {gpc.get_local_rank(ParallelMode.PIPELINE)}:{models}')