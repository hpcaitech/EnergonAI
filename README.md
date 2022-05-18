<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/165214566-467a1748-5987-4664-b5b2-d6e3367bb1b9.png" width = "600" height = "200
    " alt="Architecture" align=center />
</div>

# Energon

![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/ColossalAI-Inference/blob/main/LICENSE)


A Large-scale Model Inference System.
Energon provides 3 levels of abstraction for enabling the large-scale model inference:
- Runtime - distributed operations and customized CUDA kernels 
- Engine - encapsulate the distributed multi-device execution with the remote procedure call.
- Serving - batching requests, managing engines.
  
At present, we pre-build distributed Bert and GPT models.

For models trained by [Colossal-AI](https://github.com/hpcaitech/ColossalAI), it can be seamlessly transferred to Energon.
For single-device models, there still requires manual coding works to introduce tensor parallel and pipeline parallel.


### Installation
--- 
``` bash
$ git clone https://github.com/hpcaitech/ColossalAI-Inference.git
$ python setup.py install
```

### Large-scale Model Inference
GPT-175B


Bert-175B

Google reported a [super-large Bert (481B)](https://mlcommons.org/en/training-normal-11/) in MLPerf-Training v1.1 open, here we produce a 175B bert for displaying the performance.


<!-- ``` bash
# To pack the distributed inference as a service, we rely on Triton python backend.
$ docker run --gpus all --name=triton_server -v /<host path>/workspace:/opt/tritonserver/host --shm-size=1g --ulimit memlock=-1 -p 10010:8000 -p 10011:8001 -p 10012:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:21.10-py3

$ git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>

$ mv /examples/energon /opt/tritonserver/python_backend/models

$ bash run_gpt.sh
``` -->
### Huggingface GPT2 Generation Task Case

``` bash
# Download checkpoint
$ wget https://huggingface.co/gpt2/blob/main/pytorch_model.bin
# Download files for tokenizer
$ wget https://huggingface.co/gpt2/blob/main/tokenizer.json
$ wget https://huggingface.co/gpt2/blob/main/vocab.json
$ wget https://huggingface.co/gpt2/blob/main/merges.txt

# Launch the service
export PYTHONPATH=~/ColossalAI-Inference/examples/hf_gpt2
energon service init --config_file=~/ColossalAI-Inference/hf_gpt2/hf_gpt2_config.py

# Request for the service
Method 1: 
    FastAPI provides an automatic API docs, you can forward 
    http://127.0.0.1:8005/docs and make request with the graphical interface.
Method 2:
    curl -X 'GET' \
    'http://127.0.0.1:8005/run_hf_gpt2/I%20do%20not?max_seq_length=16' \
    -H 'accept: application/json'
```




### Contributing

If interested in making your own contribution to the project, please refer to [Contributing](./CONTRIBUTING.md) for guidance.

Thanks so much!

### Technical Overview

<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/158764528-c14538f4-8d9a-4bc8-8c6f-2e1ea82ecb59.png" width = "500" height = "350" alt="Architecture" align=center />
</div>

![图片2](https://user-images.githubusercontent.com/12018307/168971629-6df3232b-85a7-43ce-95df-f067e7e5959c.png)
![8072cc71-5a6d-4b05-a2ec-11e1ca5d76b4](https://user-images.githubusercontent.com/12018307/168971637-ffd1d6ba-44bb-4043-a275-3dc2a008c048.png)
![image (1)](https://user-images.githubusercontent.com/12018307/168971641-aebe986a-7e9d-4c66-9ced-4e8b7a1628e2.png)
![batch drawio](https://user-images.githubusercontent.com/12018307/168971644-35393802-7d8b-4e13-9428-340f7328616c.png)

