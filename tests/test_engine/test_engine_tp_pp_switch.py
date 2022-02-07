import os
import torch 
from torch import nn
import torch.multiprocessing as mp
from functools import partial
from energon.nn import Linear1D_Col, Linear1D_Row
from energon.engine import InferenceEngine

# import pytest
# torchrun --nproc_per_node=4 test_engine_tp_pp_switch.py

class BertMLP_1D(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_0 = Linear1D_Col(hidden_size, 4*hidden_size)
        # self.layer_1 = Linear1D_Row(4*hidden_size, hidden_size, parallel_input=True)

        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout ignored
    
    def forward(self, input_tensor):
        hidden_states = self.layer_0(input_tensor)
        # hidden_states = self.layer_1(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states+input_tensor)        
        return hidden_states

def check_engine_tp_pp_switch(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(4)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    config = {'hidden_size':768}

    engine = InferenceEngine(BertMLP_1D, config, pp_init_size = 2, tp_init_size = 2) # the model cannot support pipeline parallel now

    input = torch.randn(768).cuda()

    output_2_2 = engine(input)
    assert output_2_2.size(dim=-1) == 1536, "init wrong."
    
    # print(output.size())
    # print(output.device)

    engine.switch(4,1)

    output_4_1 = engine(input)
    assert output_4_1.size(dim=-1) == 3072, "switch wrong."

    # print(output.size())
    # print(output.device)

def test_engine_tp_pp_switch():
    world_size = 4
    run_func = partial(check_engine_tp_pp_switch, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)
    

if __name__ == '__main__':
    test_engine_tp_pp_switch()