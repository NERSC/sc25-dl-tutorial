#!/bin/bash

image=nersc/pytorch:25.06.01
dp=1
tp=1
cp=1
nodes=1

# parse args
for arg in "$@"
do
    if [[ $arg == tp=* ]]; then
        tp="${arg#*=}"
    elif [[ $arg == cp=* ]]; then
        cp="${arg#*=}"
    elif [[ $arg == dp=* ]]; then
        dp="${arg#*=}"
    elif [[ $arg == nodes=* ]]; then
        nodes="${arg#*=}"
    fi
done

ngpu_per_node=$(( (${tp} * ${cp} * ${dp})/$nodes ))
export MASTER_ADDR=$(hostname)
srun --nodes $nodes --ntasks-per-node $ngpu_per_node --gpus-per-node $ngpu_per_node -u shifter --image=$image --module=gpu,nccl-plugin \
    bash -c "
    source export_DDP_vars.sh
    export NVIDIA_TF32_OVERRIDE=0
    export TP=${tp}
    export CP=${cp}
    python -m pytest -s tests/test_all.py
    "
