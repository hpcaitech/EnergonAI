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

export DATASET=/data2/users/lczht/bloom-560m 
CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 

# param list
# --name :Name Path
# --tp: (int) GPU_NUM, default=1
# --http_host: (x.x.x.x)  your IP address, default=0.0.0.0
# --http_port: (xxxx) your port, default=7070
# --dtype:(str) use int8-quant or not ["fp16", "int8"], default="fp16"
# --max_batchsize:(int) limitation of batchsize, default=1
# --random_init:(bool) random init or not(if you don't have whole model data), default=False
# --random_model_size:(str) size of random init model,["560m", "7b1", "175b"],default="560m"


python server.py --tp ${GPU_NUM} --name ${DATASET}  --dtype "int8" --max_batch_size 4 --random_model_size "7b1" --random_init True


