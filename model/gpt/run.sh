#!/bin/bash

torchrun --nproc_per_node=4  evaluate.py --fp16 --model_name=gpt2_exlarge --tensor_para_size=2 --pipe_para_size=2
