#!/usr/bin/env bash

# Distributed training script for GasTwinFormer
# Usage: bash tools/dist_train.sh CONFIG NUM_GPUS [OPTIONAL_ARGS]
# Example: bash tools/dist_train.sh gastwinformer/configs/gastwinformer_80k.py 4 --amp

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

shift 2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    ${@:3}

