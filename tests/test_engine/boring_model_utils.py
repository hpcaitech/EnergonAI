from energonai.testing import BoringModel, get_correct_output
from energonai import launch_engine
from colossalai.utils import free_port
import torch
import asyncio


def run_boring_model(tp_world_size: int, pp_world_size: int):
    engine = launch_engine(tp_world_size, pp_world_size, 'localhost', free_port(), free_port(), BoringModel)
    x = torch.ones(4)
    correct_output = get_correct_output(x, pp_world_size)
    engine.submit(0, x)
    output = asyncio.run(engine.wait(0))
    try:
        assert torch.equal(output, correct_output), f'output: {output} vs target: {correct_output}'
    finally:
        engine.shutdown()
