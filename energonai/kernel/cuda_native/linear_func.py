import torch
import importlib

try:
    energonai_linear = importlib.import_module("energonai_linear_func")
except ImportError:
    raise RuntimeError('energonai_linear_func requires cuda extensions')

def EnergonLinearFunc():
    return energonai_linear.EnergonLinear()
