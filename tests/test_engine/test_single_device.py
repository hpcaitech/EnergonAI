from colossalai.testing import rerun_if_address_is_in_use
from test_engine.boring_model_utils import run_boring_model
import pytest


@pytest.mark.dist
@pytest.mark.standalone
@rerun_if_address_is_in_use()
def test_single_device():
    run_boring_model(1, 1)


if __name__ == '__main__':
    test_single_device()
