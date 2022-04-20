import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn


from energon.utils import get_timers
from example.gpt.gpt import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from energon.logging import get_dist_logger
from functools import partial
import re
import energon
from energon.engine import InferenceEngine
dtype_ = torch.float
model_config = {'dtype': dtype_,
                "checkpoint": True,
                "checkpoint_path": "GPT2.bin",
                "prefix": "",
                "vocab_size": 50257,
                "HuggingFace": True}
print(energon.__path__)
engine = InferenceEngine(model_class=gpt2_small,
                         model_config=model_config,
                         model_type='gpt',
                         max_batch_size=32,
                         tp_init_size=2,
                         pp_init_size=2,
                         port=29501,
                         dtype=dtype_)