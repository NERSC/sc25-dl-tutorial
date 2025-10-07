#!/bin/bash
DATADIR=/pscratch/sd/s/shas1693/data/sc25_tutorial_data
LOGDIR=/global/homes/s/shas1693/codes/sc25-dl-tutorial/interactive-logs
mkdir -p ${LOGDIR}
args="${@}"

export MASTER_ADDR=$(hostname)
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

set -x
srun -u -N 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 shifter --image=nersc/pytorch:24.10.01 --module=gpu,nccl-plugin -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "
#srun -u -N 1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 shifter --image=nersc/pytorch:24.08.01 -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
#    bash -c "
#    source export_DDP_vars.sh
#    ${PROFILE_CMD} python test_allgather.py ${args}
#    "
