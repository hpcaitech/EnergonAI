# the directory contains the checkpoint
export CHECKPOINT_DIR="/data/user/lclhx/opt-30B"
# the ${CONFIG_DIR} must contain a server.sh file as the entry of service
export CONFIG_DIR="/home/lcfjr/codes/EnergonAI/examples/opt"

docker run --gpus all  --rm -it -p 8090:8020 -v ${CHECKPOINT_DIR}:/model_checkpoint -v ${CONFIG_DIR}:/config --ipc=host energonai:lastest
