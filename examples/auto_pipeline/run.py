from bert import bert_small, bert_large, bert_xl, bert_8B, bert_175B
# from bert import BertEmbedding1D, BertTransformerLayer1D
from colossalai import launch_from_torch
from energonai.pipelinable import split_transformer_into_partitions

config = dict(parallel=dict(pipeline=dict(size=2), tensor=dict(size=1, mode='1d')))
launch_from_torch(config)


from energonai.context import mcfg
mcfg.load_config("/home/lcdjs/EnergonAI/examples/auto_pipeline/bert_config.py")
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

import torch.fx

batch_size = 4 
seq_len = 128

def filter_inputs(traced: torch.fx.GraphModule):
    inputs = {}
    for node in traced.graph.nodes:
        if node.op == 'placeholder':
            inputs[node.name] = None
        else:
            break

    return inputs

if gpc.get_local_rank(ParallelMode.GLOBAL) == 0:
    submodules = split_transformer_into_partitions(bert_large)

    # print(submodules.code)

    # for i in enumerate(submodules.children()):
    #     print(i)
    model0 = submodules.get_submodule('submod_0')#.graph #.print_tabular()

    model1 = submodules.get_submodule('submod_1')#.graph #.print_tabular()

    print(filter_inputs(model0))
    print(filter_inputs(model1))

    

    # print(len(submodules.children()))

    # input_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.int64).cpu()
    # attention_mask = torch.randint(0, 1, (batch_size, 1, seq_len), dtype=torch.int64).cpu()

    # # model = submodules.submod_0
    # model0 = model0.cpu()
    # model1 = model1.cpu()
    # # # print(model.parameters().device)
    # output = model0(input_ids = input_ids, attention_mask=attention_mask)

    # output = model1(blocks_11 = output, attention_mask=attention_mask)

    # submodules.to_folder("/home/lcdjs/EnergonAI/examples/auto_pipeline")
    # print(submodules.submod_1.code)

# model2 = submodules.submod_0()

# model_config = {'dtype': torch.half}

# model = bert_large(**model_config)

# graph = EnergonTracer().trace(model)
# traced = torch.fx.GraphModule(model, graph)
# print(traced.code)
