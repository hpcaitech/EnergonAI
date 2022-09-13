#!/bin/bash
set -e

export PYTHONPATH=$(realpath $(dirname $0))

find . -name "test*.py" -print0 | xargs -0L1 pytest -m "standalone"

