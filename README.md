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

### Contributing

If interested in making your own contribution to the project, please refer to [Contributing](./CONTRIBUTING.md) for guidance.

Thanks so much to all of our amazing contributors!

### Technical Overview

<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/158764528-c14538f4-8d9a-4bc8-8c6f-2e1ea82ecb59.png" width = "500" height = "350" alt="Architecture" align=center />
</div>
