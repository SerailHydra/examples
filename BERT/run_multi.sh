
PROFILE_DIR='./profile'

if [ -d ${PROFILE_DIR} ]; then
    rm -r ${PROFILE_DIR}
fi
mkdir ${PROFILE_DIR}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

nvprof --export-profile bert_4g_%p.nvvp -f --profile-child-processes \
python3.6 -m multiproc -u run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --profile

python ../merge_log_to_json.py --no-cout
