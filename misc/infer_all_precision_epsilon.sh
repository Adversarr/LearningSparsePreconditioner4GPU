EXP_NAME=$1
PRETRAINED=$2

echo "Running inference for experiment: $EXP_NAME with pretrained model: $PRETRAINED"

# for accuracy in 1e-1 1e-2 1e-4 1e-6 1e-8
# for accuracy in 1e-8
for epsilon in 3e-5 3e-4 3e-2
do
    python infer.py \
        exp_name=$EXP_NAME \
        pretrained=downloaded/heat_tetmesh-epsilon-$epsilon.ckpt \
        +out_dir=out/train_epsilon_$epsilon \
        +rtol=1e-8 \
        "${@:3}"
done