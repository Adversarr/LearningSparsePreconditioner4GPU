#!/bin/bash
ENVFILE="$(dirname "$0")/../.venv/"
source "$ENVFILE/bin/activate"
echo "TRAIN: stretch_armadillo"
echo ">>> ENV=$(realpath $ENVFILE)"
echo ">>> PYTHON=$(which python)"
echo ">>> ARGS=$@"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python train.py --config-name=basic_multidata data.block_size=3 \
    exp_name=elast_twist_remesh-unstructured \
    $@