#!/bin/bash
ENVFILE="$(dirname "$0")/../.venv/"
source "$ENVFILE/bin/activate"
echo "TRAIN: heat_tetmesh"
echo ">>> ENV=$(realpath $ENVFILE)"
echo ">>> PYTHON=$(which python)"
echo ">>> ARGS=$@"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python \
    train.py \
    exp_name=heat_tetmesh \
    data.is_fixed_topology=false \
    data.has_shared_features=false \
    $@
