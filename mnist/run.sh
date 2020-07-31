PREFIX="python3.6 -u "
SUFFIX='--epochs 1'

for var in "$@"
do
    if [ $var = "cupti" ]; then
        SUFFIX="${SUFFIX} --cupti"
    fi
    if [ $var = "nsight" ]; then
        SUFFIX="${SUFFIX} --nsight"
        PREFIX="/opt/nvidia/nsight-systems/2020.3.1/bin/nsys profile ${PREFIX}"
    fi
done

CUDA_VISIBLE_DEVICES=1

$PREFIX main.py $SUFFIX
