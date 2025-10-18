#!/usr/bin/env bash

# Distributed testing script for GasTwinFormer
# Usage: bash tools/dist_test.sh CONFIG CHECKPOINT NUM_GPUS [OPTIONAL_ARGS]
# Example: bash tools/dist_test.sh gastwinformer/configs/gastwinformer_80k.py checkpoints/model.pth 4

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

shift 3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

