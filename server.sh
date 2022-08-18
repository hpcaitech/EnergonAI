#!/usr/bin/env bash

set -xe
cd $(dirname $0)

export BASE=./examples/opt

export PYTHONPATH=${BASE}
energonai service init --config_file=${BASE}/opt_config.py
