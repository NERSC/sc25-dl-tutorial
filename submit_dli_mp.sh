#!/bin/bash 

DATADIR=/data
LOGDIR=/dli/task/logs
mkdir -p ${LOGDIR}
args="${@}"


# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

set -x
mpirun --allow-run-as-root -np 2  \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "
