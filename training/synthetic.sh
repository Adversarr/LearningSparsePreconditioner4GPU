#!/bin/bash
ENVFILE="$(dirname "$0")/../.venv/"
source "$ENVFILE/bin/activate"
echo "TRAIN: synthetic"
echo ">>> ENV=$(realpath $ENVFILE)"
echo ">>> PYTHON=$(which python)"
echo ">>> ARGS=$@"
echo ">>> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python train.py \
    exp_name=synthetic \
    data.is_fixed_topology=false \
    data.has_shared_features=false \
    data.use_node_features=false \
    data.use_edge_features_as_node_feature=mean \
    $@
