import os
import torch
import torch.multiprocessing as mp
from functools import partial
from energon.engine import InferenceEngine

import sys
sys.path.append('/home/lcdjs/ColossalAI-Inference-tmp/model/gpt')
from model.pipeline_gpt1d import GPT2_small_pipeline_1D, GPT2_exlarge_pipeline_1D, GPT3_pipeline_1D

def check_engine_tp_pp_switch(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(4)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    config = {'num_chunks':1, 'checkpoint':False, 'dtype':torch.float, 'embed_split_hidden':False}

    input_ids = torch.randint(1, 10, (4, 1024), dtype=torch.int64)
    attention_mask = torch.randint(0, 1, (4, 1, 1024), dtype=torch.int64)
    hidden_states = None
    sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)

    engine = InferenceEngine(GPT2_small_pipeline_1D, config, sample, pp_init_size = 4, tp_init_size = 1)

    

    output_2_2 = engine.run()
    
    
    # print(output.size())
    # print(output.device)

    engine.switch(4,1)

    output_4_1 = engine.run()

    # print(output.size())
    # print(output.device)

def test_engine_tp_pp_switch():
    world_size = 4
    run_func = partial(check_engine_tp_pp_switch, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)
    

if __name__ == '__main__':
    test_engine_tp_pp_switch()
