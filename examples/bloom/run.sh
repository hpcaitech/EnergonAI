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

<<<<<<< HEAD
export GPU_NUM=1
CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 

torchrun --standalone --nproc_per_node=${GPU_NUM} accelerate_script.py --name '/data2/users/lczht/bloom-560m'
=======
export GPU_NUM=2
CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 

python accelerate_script.py --tp ${GPU_NUM} --name /data2/users/lczht/bloom-560m --cache_size 0
>>>>>>> origin
