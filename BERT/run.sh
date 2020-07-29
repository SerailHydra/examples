
PROFILE_DIR='./profile'

if [ -d ${PROFILE_DIR} ]; then
    rm -r ${PROFILE_DIR}
fi
mkdir ${PROFILE_DIR}

rm *.nvvp
rm *.log

PREFIX="python3.6 -u "
SUFFIX=
single_gpu=1
nvprof_on=0
multi=0
dist=0

for var in "$@"
do
    if [ $var = "cupti" ]; then
        SUFFIX="${SUFFIX} --cupti"
    fi
    if [ $var = "multi" ]; then # run multi-GPU profile
        single_gpu=0
        multi=1
        PREFIX="${PREFIX}-m multiproc "
    fi
    if [ $var = "nvprof" ]; then # use nvprof, cannot coexist with cupti
        nvprof_on=1
    fi
    if [ $var = "nsight" ]; then # turn on NeuralTap profile
        SUFFIX="${SUFFIX} --nsight"
        PREFIX="/opt/nvidia/nsight-systems/2020.3.1/bin/nsys profile ${PREFIX}"
    fi
    if [ $var = "dist" ]; then
        single_gpu=0
        dist=1
        #export NCCL_SOCKET_IFNAME=enp3s0f0
        #export MASTER_ADDR=eco-11
        #export MASTER_PORT=23451
        PREFIX="${PREFIX}-m dist_multiproc "
        SUFFIX="${SUFFIX} --dist-url env://"
    fi
done

if [ $nvprof_on = 1 ] && [ $single_gpu = 0 ]; then
    if [ $multi = 1 ]; then
        PREFIX="nvprof --export-profile bert_multi_%p.nvvp -f --profile-child-processes ${PREFIX}"
    fi
    if [ $dist = 1 ]; then
        PREFIX="nvprof --export-profile bert_dist_%p.nvvp -f --profile-child-processes ${PREFIX}"
    fi
elif [ $nvprof_on = 1 ]; then
    PREFIX="nvprof --export-profile bert.nvvp -f ${PREFIX}"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

if [ $single_gpu = 1 ]; then
    echo "using single GPU"
    export CUDA_VISIBLE_DEVICES=1
fi

#export CUDA_VISIBLE_DEVICES=0,1

$PREFIX run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_lower_case \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ $SUFFIX
