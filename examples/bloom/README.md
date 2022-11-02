# Energon-AI for Bloom inference
To start service
```
bash run.sh
```
While the service is running, send `POST` requests to https://[ip]:[port]/generation     

send POST body as json file:

```
curl -X POST https://[ip]:[port]/generation -H "Content-Type: application/json" -d '{"prompt": "However, there are still immense benefits to learning quantum computing.", "max_new_tokens": 60}'
```
received message: 
```
{
    "text": [
        "Four years after her last encounter with masked killer Michael Myers, Laurie Strode is living with her mother, who is now a widow. Her mother, who is also a widow, is a member of the New York City police force. Myers is a member of the New York City police force, and he is a member of the New York City police force. Myers is a member of the"
    ]
}
```