CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

export GPU_NUM=2
export DATASET=/data2/users/lczht/bloom-560m # /data2/users/lczht/bloom-560m 
                                            #/data2/users/lccsr/bloom3b/data 
                                            #/data2/users/lccsr/bloom1b7/data
CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 

export USE_INT8=1

if [[ ${USE_INT8} == 1 ]]; then
DTYPE="int8"
else
DTYPE="fp16"
fi

python server.py --tp ${GPU_NUM} --name ${DATASET} --dtype ${DTYPE} --max_batch_size 4
