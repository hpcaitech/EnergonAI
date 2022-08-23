#!/usr/bin/env bash

set -xe
cd $(dirname $0)

export BASE=${PWD}

export PYTHONPATH=${BASE}
energonai service init --config_file=${BASE}/opt_gen_config.py
