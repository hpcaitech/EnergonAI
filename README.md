![vighnesh-dudani-ZQSs0YZUNfA-unsplash](https://user-images.githubusercontent.com/12018307/165212624-c0c98042-f111-48f8-95a6-c318e08dc57f.png)

# ColossalAI-Inference

[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/FastFold/blob/main/LICENSE)

Temporary repo for large-scale model inference.


### Installation
--- 
``` bash
$ git clone https://github.com/hpcaitech/ColossalAI-Inference.git
$ python setup.py install or python setup.py develop
```

### Quick Start
---
``` bash
# To pack the distributed inference as a service, we rely on Triton python backend.
$ docker run --gpus all --name=triton_server -v /<host path>/workspace:/opt/tritonserver/host --shm-size=1g --ulimit memlock=-1 -p 10010:8000 -p 10011:8001 -p 10012:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:21.10-py3

$ git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>

$ mv /examples/energon /opt/tritonserver/python_backend/models

$ bash run_gpt.sh
```
### Huggingface GPT2 Demo

<img width="1073" alt="30826ccd5ab94a2a14ba166132d780c" src="https://user-images.githubusercontent.com/12018307/164587795-6f70a473-ac87-47e2-bb91-926fc6b182ba.png">


### Contributing

If interested in making your own contribution to the project, please refer to [Contributing](./CONTRIBUTING.md) for guidance.

Thanks so much to all of our amazing contributors!

### Technical Overview

<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/158764528-c14538f4-8d9a-4bc8-8c6f-2e1ea82ecb59.png" width = "500" height = "350" alt="Architecture" align=center />
</div>
