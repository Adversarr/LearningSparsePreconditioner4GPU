#!/bin/bash
ENVFILE="$(dirname "$0")/../.venv/"
source "$ENVFILE/bin/activate"
echo "TRAIN: elast_twist"
echo ">>> ENV=$(realpath $ENVFILE)"
echo ">>> PYTHON=$(which python)"
echo ">>> ARGS=$@"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python \
    train.py \
    exp_name=elast_twist \
    data.block_size=3 \
    $@
