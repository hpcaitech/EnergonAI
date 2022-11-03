# Energon-AI for Bloom inference
# How to run
To start service
```
bash run.sh
```
While the service is running, send `POST` requests to https://[ip]:[port]/generation     

send POST body as json file:
```
curl -X POST http://ip:port/generation -H "Content-Type: application/json" -d @test.json
```

test.json looks like follows:
```
{
    "prompt": "However, there are still immense benefits to learning quantum computing.",
    "top_p": 0.90,
    "top_k": 40,
    "max_new_tokens": 60
}
```  

received message: 
```
{
    "text": "However, there are still immense benefits to learning quantum computing. For example, quantum computing can be used to solve problems that are difficult to solve by classical methods. Quantum computing can also be used to solve problems that are difficult to solve by classical methods. Quantum computing can also be used to solve problems that are"
}
```

# Configure
## Configure batching
add `--max_batch_size <MaxBatchSize>`  to the python command in `run.sh`

The `<MaxBatchSize>` can be an integer in `[1, MAXINT]`. The engine will make batch whose size is less or equal to this value.

Bigger MaxBatchSize may speed up concurrent requests in a single batched forwarding process.
