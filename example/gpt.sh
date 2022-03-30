#!/bin/bash

tp_size=2
pp_size=2
model=gpt2_large
world_size=`expr $tp_size \* $pp_size`
server_port_start=8005
host="localhost"
port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH=/home/lcdjs/ColossalAI-Inference/example

for ((i=1; i<${world_size}; i++))  
do
server_port=`expr $server_port_start + $i`
uvicorn server:app --app-dir /home/lcdjs/ColossalAI-Inference/energon/engine/ --port ${server_port} &
echo "process: ${i} launches"
done

sleep 3

for ((i=1; i<${world_size}; i++))  
do
server_port=`expr $server_port_start + $i`
curl -X 'GET' \
"http://127.0.0.1:${server_port}/start/${tp_size}?pp_size=${pp_size}&backend=nccl&seed=1024&verbose=true&rank=${i}&local_rank=${i}&host=${host}&port=${port}" \
-H 'accept: application/json' &
echo "http://127.0.0.1:${server_port}/start/${tp_size}?pp_size=${pp_size}&backend=nccl&seed=1024&verbose=true&rank=${i}&local_rank=${i}&host=${host}&port=${port}"
echo "evoke process: ${i} init rpc"
done

python3  gpt_inference.py --fp16 --model_name=${model} --tensor_para_size=${tp_size} --pipe_para_size=${pp_size} --port=${port}
# tritonserver --model-repository /opt/tritonserver/host/python_backend/models
