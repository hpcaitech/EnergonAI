#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python  evaluate.py --fp16 --model_name=gpt2_exlarge --tensor_para_size=2 --pipe_para_size=2
