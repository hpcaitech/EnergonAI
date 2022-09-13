from energonai.testing import BoringModel, get_correct_output
from colossalai.testing import rerun_if_address_is_in_use
from energonai import launch_engine
from colossalai.utils import free_port
import pytest
import torch
import asyncio


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_hybrid():
    engine = launch_engine(2, 2, 'localhost', free_port(), free_port(), BoringModel)
    x = torch.ones(4)
    correct_output = get_correct_output(x, 2)
    engine.submit(0, x)
    output = asyncio.run(engine.wait(0))
    try:
        assert torch.equal(output, correct_output)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    test_hybrid()
