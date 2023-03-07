<div  align="center">
    <img src="https://user-images.githubusercontent.com/12018307/170214566-b611b131-fff1-41c0-9447-786a8a6f0bac.png" width = "600" height = "148" alt="Architecture" align=center />
</div>

# Energon-AI

![](https://img.shields.io/badge/Made%20with-ColossalAI-blueviolet?style=flat)
[![GitHub license](https://img.shields.io/github/license/hpcaitech/FastFold)](https://github.com/hpcaitech/ColossalAI-Inference/blob/main/LICENSE)

A service framework for large-scale model inference, Energon-AI has the following characteristics:

- **Parallelism for Large-scale Models:** With tensor parallel operations, pipeline parallel wrapper, distributed checkpoint loading, and customized CUDA kernel, EnergonAI can enable efficient parallel inference for larges-scale models.
- **Pre-built large models:** There are pre-built implementation for popular models, such as OPT. It supports the cache technique for the generation task and distributed parameter loading.
- **Engine encapsulation：** There has an abstraction layer called engine. It encapsulates the single instance multiple devices (SIMD) execution with the remote procedure call, making it acts as the single instance single device (SISD) execution.
- **An online service system:** Based on FastAPI, users can launch a web service of the distributed infernce quickly. The online service makes special optimizations for the generation task. It adopts both left padding and bucket batching techniques for improving the efficiency.

For models trained by [Colossal-AI](https://github.com/hpcaitech/ColossalAI), they can be easily transferred to Energon-AI.
For single-device models, they require manual coding works to introduce tensor parallelism and pipeline parallelism.


## Installation

There are three ways to install energonai.

- **Install from pypi**

``` bash
pip install energonai
```


- **Install from source**
``` bash
$ git clone git@github.com:hpcaitech/EnergonAI.git
$ pip install -r requirements.txt
$ pip install .
```

- **Use docker**
``` bash
$ docker pull hpcaitech/energon-ai:latest
```


## Build an online OPT service in 5 minutes

1. **Download OPT model:**
To launch the distributed inference service quickly, you can download the checkpoint of OPT-125M [here](https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt). You can get details for loading other sizes of models [here](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt/script).

2. **Launch an HTTP service:**
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
    Now, we can launch a service:

    ```bash
        bash server.sh
    ```

    Then open ***https://[ip]:[port]/docs*** in your browser and try out!


## Publication
You can find technical details in our blog and manuscript:

[Build an online OPT service using Colossal-AI in 5 minutes](https://www.colossalai.org/docs/advanced_tutorials/opt_service/)

[EnergonAI: An Inference System for 10-100 Billion Parameter Transformer Models](https://arxiv.org/pdf/2209.02341.pdf)

```
@misc{du2022energonai, 
      title={EnergonAI: An Inference System for 10-100 Billion Parameter Transformer Models}, 
      author={Jiangsu Du and Ziming Liu and Jiarui Fang and Shenggui Li and Yongbin Li and Yutong Lu and Yang You},
      year={2022},
      eprint={2209.02341},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contributing

If interested in making your own contribution to the project, please refer to [Contributing](./CONTRIBUTING.md) for guidance.

Thanks so much!
