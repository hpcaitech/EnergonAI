import requests
import json

url = "http://localhost:7070/generation"  
data = {
    "max_tokens": 200,
    "prompt": "Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:"
}
response = requests.post(url, json=data)
print(response.json())
