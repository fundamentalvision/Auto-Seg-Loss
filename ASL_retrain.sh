#!/usr/bin/env bash

set -x


CODEBASE="$(dirname $0)"
CONFIGNAME=$1
GPUS=${2-4}
SEED=${3-1}

CONFIG=${CODEBASE}/ASL_configs/retrain/${CONFIGNAME}


MAINFILE="tools/train.py"

PYTHONPATH=$CODEBASE:$PYTHONPATH \
python -u -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT-12345} \
    ${CODEBASE}/${MAINFILE} \
    ${CONFIG} \
    --launcher="pytorch" \
    --work-dir ${CODEBASE}/exp/retrain/${CONFIGNAME}/ \
    --seed ${SEED}
    