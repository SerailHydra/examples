PREFIX="python3.6 -u "
SUFFIX='--epochs 1'

for var in "$@"
do
    if [ $var = "nsight" ]; then
        PREFIX="/opt/nvidia/nsight-systems/2020.3.1/bin/nsys profile ${PREFIX}"
    fi
done

$PREFIX main.py $SUFFIX
