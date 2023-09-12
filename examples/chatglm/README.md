# Overview

This is an example showing how to run ChatGLM generation. The ChatGLM model is implemented using ColossalAI.

It supports tensor parallelism, batching and caching.

# How to run

Run ChatGLM-6B:
```shell
python examples/chatglm/ChatGLM_fastapi.py glm-6b --checkpoint /data2/zxy/qkv_4_energon_glm --master_port 19991 --rpc_port 19981 --tp 2 --queue_size 12
```
```shell
python examples/chatglm/test_glm_api.py
```

It will launch a HTTP server on `0.0.0.0:7071` by default and you can customize host and port. You can open `localhost:7070/docs` in your browser to see the openapi docs.

## Configure

### Configure model
```shell
python ChatGLM_fastapi.py <model>
```
Available models: ChatGLM-6B.

### Configure tensor parallelism
```shell
python ChatGLM_fastapi.py <model> --tp <TensorParallelismWorldSize>
```
The `<TensorParallelismWorldSize>` can be an integer in `[1, #GPUs]`. Default `1`.

### Configure checkpoint
```shell
python ChatGLM_fastapi.py <model> --checkpoint <CheckpointPath>
```
The `<CheckpointPath>` can be a file path or a directory path. If it's a directory path, all files under the directory will be loaded.

### Configure queue
```shell
python ChatGLM_fastapi.py <model> --queue_size <QueueSize>
```
The `<QueueSize>` can be an integer in `[0, MAXINT]`. If it's `0`, the request queue size is infinite. If it's a positive integer, when the request queue is full, incoming requests will be dropped (the HTTP status code of response will be 406).

### Configure bathcing
```shell
python ChatGLM_fastapi.py <model> --max_batch_size <MaxBatchSize>
```
The `<MaxBatchSize>` can be an integer in `[1, MAXINT]`. The engine will make batch whose size is less or equal to this value.

Note that the batch size is not always equal to `<MaxBatchSize>`, as some consecutive requests may not be batched.

### Configure caching
```shell
python ChatGLM_fastapi.py <model> --cache_size <CacheSize> --cache_list_size <CacheListSize>
```
This will cache `<CacheSize>` unique requests. And for each unique request, it cache `<CacheListSize>` different results. A random result will be returned if the cache is hit.

The `<CacheSize>` can be an integer in `[0, MAXINT]`. If it's `0`, cache won't be applied. The `<CacheListSize>` can be an integer in `[1, MAXINT]`.

### Other configurations
```shell
python ChatGLM_fastapi.py -h
```

# Pre-process pre-trained weights
## ChatGLM-6B
Convert chatglm weights on huggingface and store them using th folling script 

See [script/processing_ckpt_6b.py](./script/processing_ckpt_6b.py).

# conda环境
gcc/gxx 版本>7.2

cuda>= 11.7

pytorch>=1.13.1

# developer introduce
中图科信 http://www.cnpeak.com/
产品 https://note.kxsz.net/
