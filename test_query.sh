set -x

# Launch a docker container in this way
# docker run --gpus all  --rm -it -p 8090:8020 -v /data:/data --ipc=host energonai:lastest

# run this script in the host machine.
#curl -X 'GET' \
 #   'http://127.0.0.1:8090/run/It%27s%20my%20turn?max_seq_length=200' \
  #  -H 'accept: application/json'
curl -X 'POST' \
  'http://a6b43542b93ed44779bc240c6eb10bac-255560469.us-east-1.elb.amazonaws.com/generation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "max_tokens": 10,
  "prompt": "what is your name?",
  "top_k": 10,
  "top_p": 10,
  "temperature": 0.8
}'
