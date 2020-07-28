#!/bin/bash

set -e

DATASET_DIR='/mnt/dataset/wmt_ende/'
RESULTS_DIR='gnmt_wmt16'
PROFILE_DIR='./profile'

SEED=${1:-"1"}
TARGET=${2:-"21.80"}

if [ -d ${PROFILE_DIR} ]; then
    rm -r ${PROFILE_DIR}
fi
mkdir ${PROFILE_DIR}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# run training
#nvprof --export-profile seq2seq_4g.nvvp \
python -u -m multiproc train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --epochs 8 \
  --batch-size 64 \
  --profile

python ../../merge_log_to_json.py
