# Energon-AI for Bloom inference
To start service
```
bash run.sh
```
While the service is running, send `POST` requests to https://[ip]:[port]/generation     

send POST body as json file:
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