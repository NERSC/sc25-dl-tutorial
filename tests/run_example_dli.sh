#!/bin/bash 

tp=${1}

export MASTER_ADDR=$(hostname)

set -x
mpirun --allow-run-as-root -np 2  \
    bash -c "
    source export_DDP_vars.sh
    python -m tests.make_mlp_tensor_par --tp ${tp}
    "
