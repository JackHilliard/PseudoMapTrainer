#!/usr/bin/env bash
# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

CONFIG=$1
GPUS=$2
PORT=${PORT:-$(( (RANDOM * 32768 + RANDOM) % 55536 + 10000 ))} # random port between 10000 and 65535

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_pseudo_labels.py $CONFIG --launcher pytorch ${@:3} --eval chamfer
