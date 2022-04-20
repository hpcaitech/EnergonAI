import random
import numpy as np
import argparse
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from transformers import GPT2Tokenizer

from energon.context import ParallelMode
from energon.core import global_context as gpc

from energon.utils import get_timers
from example.gpt.gpt import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_8B, gpt3
from energon.logging import get_dist_logger
from functools import partial
import re

from energon.engine import InferenceEngine

MODEL_CLASSES = {
    "gpt2_small": gpt2_small,
    "gpt2_medium": gpt2_medium,
    "gpt2_large": gpt2_large,
    "gpt2_xl": gpt2_xl,
    "gpt2_8B": gpt2_8B,
    "gpt3": gpt3
}

def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index


def build_gpt_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_para_size", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument("--pipe_para_size", type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument("--port", type=int, default=29500, help="Port")
    parser.add_argument("--iteration", type=int, default=100, help="Iteration")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit precision instead of 32-bit")
    args = parser.parse_args()
    # tp_size = args.tensor_para_size
    # pp_size = args.pipe_para_size
    dtype_ = torch.float
    if args.fp16:
        dtype_ = torch.half
    model_config = {'dtype': dtype_,
                    "checkpoint": True,
                    "checkpoint_path": "GPT2.bin",
                    "prefix": "",
                    "vocab_size": 50257,
                    "HuggingFace": True}
    engine = InferenceEngine(model_class=gpt2_small,
                             model_config=model_config,
                             model_type='gpt',
                             max_batch_size=32,
                             tp_init_size=args.tensor_para_size,
                             pp_init_size=args.pipe_para_size,
                             port=args.port,
                             dtype=dtype_)
    tokenizer = GPT2Tokenizer.from_pretrained('./')
    # tokenizer = GPT2Tokenizer(vocab_file="vocab.json", merges_file="merges.txt")
    # test_input = ["MANY YEARS LATER as he faced the firing squad, Colonel Aureliano Buendía was to remember that"
    #               for _ in range(10)]
    test_input = "What the fuck"
    print(test_input)
    input_token = tokenizer(test_input, return_tensors="pt")
    # tokens_tensor = torch.tensor([input_token])
    total_predicted_text = test_input
    # print(input_ids['input_ids'])
    # print(input_ids['attention_mask'])
    # input_ids = input_token['input_ids']
    # print(input_ids.shape)
    # input_ids = torch.randint(1, 10, (32, 40), dtype=torch.int64)
    # attention_mask = torch.randint(0, 1, (32, 1, 40), dtype=torch.int64)
    hidden_states = None
    output = engine.run(input_token)
    predictions = output.to_here()
    predicted_index = select_top_k(predictions, k=1)
    total_predicted_text += tokenizer.decode(predicted_index)
    # print(total_predicted_text)
    # sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
    # sample = dict(hidden_states=hidden_states, input_ids=input_ids)

    # output = engine.run(input_token)
    # output.to_here()
    # print(tokenizer.decode(output.to_here()[0]))
    timer = get_timers()
    timer('time1').start()

    for i in range(1, args.iteration):
        output = engine.run(input_token)
        predictions = output.to_here()
        predicted_index = select_top_k(predictions, k=10)
        total_predicted_text += tokenizer.decode(predicted_index)
        # print(total_predicted_text)
        if '<|endoftext|>' in total_predicted_text:
            # 如果出现文本结束标志，就结束文本生成
            break

        input_token = tokenizer(total_predicted_text, return_tensors="pt")

        # input_ids = tokenizer(test_input, return_tensors='pt')
        # # input_ids = torch.randint(1, 10, (32, 40), dtype=torch.int64)
        # # attention_mask = torch.randint(0, 1, (32, 1, 40), dtype=torch.int64)
        # hidden_states = None
        # # sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
        # # sample = dict(hidden_states=hidden_states, input_ids=input_ids)
        # # input_ids = torch.randint(1, 10, (32, i % 20 + 2), dtype=torch.int64)
        # # attention_mask = torch.randint(0, 1, (32, 1, i % 20 + 2), dtype=torch.int64)
        # # hidden_states = None
        # # sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)
        # output = engine.run(input_ids)

        # print(tokenizer.decode(output.to_here()))
    print(total_predicted_text)
    # print(output.to_here())
    timer('time1').stop()

    time1 = timer('time1').elapsed()
    logger = get_dist_logger()

    # logger.info(f'Time0, '
    #             f'Pipeline Rank/ Tensor Rank: {args.pipe_para_size}/{gpc.get_world_size(ParallelMode.PARALLEL_1D)},'
    #             f'Time: {time0/args.iteration}')
    logger.info(f'Time1, '
                f'Pipeline Rank/ Tensor Rank: {args.pipe_para_size}/{gpc.get_world_size(ParallelMode.PARALLEL_1D)},'
                f'Time: {time1 / args.iteration}')

    engine.clear()
    #

if __name__ == "__main__":
    build_gpt_model()
