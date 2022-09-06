<!-- <div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/170214566-b611b131-fff1-41c0-9447-786a8a6f0bac.png" width = "600" height = "148" alt="Architecture" align=center />
</div> -->
# Energon-AI

![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/ColossalAI-Inference/blob/main/LICENSE)

A service framework for large-scale model inference, Energon-AI has the following characteristics:

- **Parallelism for Large-scale Models:** With tensor parallel operations, pipeline parallel wrapper, distributed checkpoint loading, and customized CUDA kernel, EnergonAI can enable efficient parallel inference for larges-scale models.
- **Pre-built large models:** There are pre-built implementation for popular models, such as OPT. It supports cache technique for the generation task and parameter loading of official checkpoints.
- **Engine encapsulation：** There has abstraction layer called engine. It encapsulates the single instance multiple devices (SIMD) execution with the remote procedure call, making it acts as the single instance single device (SISD) execution.
- **An online service system:** Based on FastAPI, users can launch a web service of a distributed infernce quickly. The online service makes special optimitions for the generation task. It adopts both left padding and bucket batching techniques for improving the efficiency.

For models trained by [Colossal-AI](https://github.com/hpcaitech/ColossalAI), they can be easily transferred to Energon-AI.
For single-device models, they require manual coding works to introduce tensor parallelism and pipeline parallelism.


### Installation
**Install from source**
``` bash
$ git clone git@github.com:hpcaitech/EnergonAI.git
$ pip install -r requirements.txt
$ pip install .
```
**Use docker**
``` bash
$ docker pull hpcaitech/energon-ai:0.2.3
```


### Build an online OPT service in 5 minutes

1. **Download OPT model:**
  To launch the distributed inference service quickly, you can download the checkpoint of OPT-125M [here](https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt). You can get details for loading other sizes of models [here](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt/script).
2. **Prepare a prebuilt service image:**
Pull a docker image from dockerhub installed with ColossalAI and EnergonAI.
    ```bash
        docker pull hpcaitech/energon-ai:0.2.3
    ```
    You can also install the colossalai and Energon-ai without a docker by following their readme pages.

3. **Launch an HTTP service:**
To launch a service, we need to provide python scripts to describe the model type and related configurations, and start an http service.
An OPT example is [EnergonAI/examples/opt](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt).  
The entrance of the service is a bash script ***server.sh***.
The config of the service is at ***opt_config.py***, which defines the model type, the checkpoint file path, the parallel strategy, and http settings. You can adapt it for your own case.
For example, set the model class as opt_125M and set the correct checkpoint path as follows. Set the tensor parallelism degree the same as your gpu number.
    ```bash
        model_class = opt_125M
        checkpoint = 'your_file_path'
        tp_init_size = #gpu
    ```
    Now, we can launch a service using docker. You can map the path of checkpoint and directory containing configs to docker disk volume.
    ```bash
        export CHECKPOINT_DIR="your_opt_checkpoint_path"
        export CONFIG_DIR="config_file_path"
        # the ${CONFIG_DIR} must contain a server.sh file as the entry of service
        docker run --gpus all  --rm -it -p 8020:8020 -v ${CHECKPOINT_DIR}:/model_checkpoint -v ${CONFIG_DIR}:/config --ipc=host energonai:lastest
    ```

    Then open ***https://[ip]:8020/docs*** in your browser and try out!


### Contributing

If interested in making your own contribution to the project, please refer to [Contributing](./CONTRIBUTING.md) for guidance.

Thanks so much!

### Publication
You can find technical details in our blog and paper:

Cite this paper, if you use EnergonAI in your research publication:



<!-- ### Large-scale Model Inference Performance
#### Scaling Ability

Here GPT3-12-layers in FP16 is adopted.  
Here a node with 8 A100 80 GB GPUs is adopted. GPUs are fully connected with NvLink.   
Energon-AI adopts the redundant computation elimination method. The method is first raised in [EffectiveTransformer](https://github.com/bytedance/effective_transformer), and our implementation refers to [TurboTransformer](https://github.com/Tencent/TurboTransformers/blob/master/turbo_transformers/layers/kernels/gpu_transpose_kernel.cu).  
Here the sequence length is set the half of the padding length.
<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/168971637-ffd1d6ba-44bb-4043-a275-3dc2a008c048.png" width = "600" height = "240" alt="Architecture" align=center />
</div>

#### Latency
Here GPT3 in FP16 is adopted.  
Here a node with 8 A100 80 GB GPUs is adopted. Every two GPUs are connected with NvLink.  
Here the sequence length is set the half of the padding length when using redundant computation elimination method, which is the Energon-AI(RM).  
Here FasterTransformer is adopted in comparison and it does not support the redundant computation elimination method in the distributed execution.
<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/169728315-8ac95e4f-3e81-44e5-b82b-5873ffe85351.png" width = "600" height = "300" alt="Architecture" align=center />
</div>

#### Batching
Energon-AI dynamically selects the batch processing with the highest priority regarding the waiting time, batch size, batch expansion possibility (based on the sentence length after padding).
Our dynamic batching method is inspired by the DP algorithm from [TurboTransformer](https://dl.acm.org/doi/10.1145/3437801.3441578).  
Here FIFO batching is selected in comparison.
<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/170616782-18fae36f-75cd-4e7b-bc0b-c8998be1e540.png" width = "400" height = "100" alt="Architecture" align=center />
</div>

### Technical Overview

<div  align="center">    
    <img src="https://user-images.githubusercontent.com/12018307/168971629-6df3232b-85a7-43ce-95df-f067e7e5959c.png" width = "480" height = "500" alt="Architecture" align=center />
</div> -->

<!-- ### Cite us
Cite this paper, if you use EnergonAI in your research publication. -->


<!-- ### Launch an http service using docker
``` bash
bash ./docker/launch.sh
``` -->


<!-- ### Huggingface GPT2 Generation Task Case

``` bash
# Download checkpoint
$ wget https://huggingface.co/gpt2/blob/main/pytorch_model.bin
# Download files for tokenizer
$ wget https://huggingface.co/gpt2/blob/main/tokenizer.json
$ wget https://huggingface.co/gpt2/blob/main/vocab.json
$ wget https://huggingface.co/gpt2/blob/main/merges.txt

# Launch the service
export PYTHONPATH=~/EnergonAI/examples/hf_gpt2
energonai service init --config_file=~/EnergonAI/hf_gpt2/hf_gpt2_config.py

# Request for the service
Method 1: 
    FastAPI provides an automatic API docs, you can forward 
    http://127.0.0.1:8005/docs and make request with the graphical interface.
Method 2:
    curl -X 'GET' \
    'http://127.0.0.1:8005/run_hf_gpt2/I%20do%20not?max_seq_length=16' \
    -H 'accept: application/json'  -->
```