#!/usr/bin/env bash

set -xe
cd $(dirname $0)

export BASE=./examples/hf_gpt2

export PYTHONPATH=${BASE}
energonai service init --config_file=${BASE}/hf_gpt2_config.py
