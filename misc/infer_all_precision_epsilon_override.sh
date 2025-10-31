EXP_NAME=$1
PRETRAINED=$2

echo "Running inference for experiment: $EXP_NAME with pretrained model: $PRETRAINED"

# for accuracy in 1e-1 1e-2 1e-4 1e-6 1e-8
# for accuracy in 1e-8
for epsilon in 1e-6 1e-4 1e-2 1e-1
do
    python infer.py \
        exp_name=$EXP_NAME \
        pretrained=$PRETRAINED \
        +out_dir=out/epsilon_$epsilon \
        +rtol=1e-8 \
        +override_epsilon=$epsilon \
        "${@:3}"
done