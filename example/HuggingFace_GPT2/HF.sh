#!/bin/bash

tp_size=2
pp_size=2
model=gpt2_small
world_size=`expr $tp_size \* $pp_size`
server_port_start=8005
host="localhost"
port=29499
CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH=/home/lcdjs/ColossalAI-Inference/example

for ((i=1; i<${world_size}; i++))
do
server_port=`expr $server_port_start + $i`
python3 /home/lclzm/ColossalAI_Inference/energon/engine/server.py --port ${server_port} &
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

python3  HF_GPT2_inference.py --fp16 --tensor_para_size=${tp_size} --pipe_para_size=${pp_size} --port=${port}
# python3  gpt_inference.py --fp16 --model_name=gpt2_large --tensor_para_size=1 --pipe_para_size=2 --port=29499
# tritonserver --model-repository /opt/tritonserver/host/python_backend/models
