set -x

# Launch a docker container in this way
# docker run --gpus all  --rm -it -p 8090:8020 -v /data:/data --ipc=host energonai:lastest /bin/bash
# bash /workspace/EnergonAI/server.sh

# run this script in the host machine.
curl -X 'GET' \
    'http://127.0.0.1:8090/run/It%27s%20my%20turn?max_seq_length=200' \
    -H 'accept: application/json'
