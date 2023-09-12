import requests
import json
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3,4'
from tqdm import tqdm
from transformers import GPT2Model , AutoModel , AutoTokenizer
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model
from transformers.testing_utils import require_torch, slow, torch_device
import numpy as np
import random
import torch

def test_gpt():
    path='/data2/share/gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2Model.from_pretrained(path)
    for name,parm in model.named_parameters():
        print(name,';',parm.size())


    text = "北京大学"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

def test_glmmodel():

    # model = GPT2Model.from_pretrained("/data2/zxy/gpt2-medium")
    # for name, param in model.named_parameters():
    #     print(name, ": ", param.size())

    # for name in model.state_dict():
    #     print(name, ":",model.state_dict()[name].shape)

    # glm_model =AutoModel.from_pretrained('/data/share/chatglm-6b/',trust_remote_code=True)
    # for name, param in glm_model.state_dict().items():
    #     print(name, ":", param.shape)
    # print('hello')
    tokenizer = AutoTokenizer.from_pretrained("/data/share/chatglm-6b/", trust_remote_code=True)
    print(tokenizer)
    model = AutoModel.from_pretrained("/data/share/chatglm-6b/", trust_remote_code=True).half().cuda()
    response, history = model.chat(tokenizer, "物理是什么", history=[])
    print(response)



def get_model_and_tokenizer():
    model = AutoModel.from_pretrained("/data/share/chatglm-6b", trust_remote_code=True).half()
    model.to(torch_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("/data/share/chatglm-6b", trust_remote_code=True)
    return model, tokenizer

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def glm_generate():
    model, tokenizer = get_model_and_tokenizer()
    sentences = [
        "今天天气大概25度，有点小雨，吹着风，我想去户外散步，应该穿什么样的衣服裤子鞋子搭配。",
    ]
    set_random_seed(42)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs = inputs.to(torch_device)

    for i in tqdm(range(100)):
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=200,
            num_beams=4
        )

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(batch_out_sentence)

def test_api():
    url = "http://localhost:7071/generation" # 修改为你的服务器地址
    data = {
        "max_tokens": 200,
        "prompt": "今天天气大概25度，有点小雨，吹着风，我想去户外散步，应该穿什么样的衣服裤子鞋子搭配。",
        "do_sample":False,
    }
    for i in tqdm(range(10)):
        response = requests.post(url, json=data)
        print(response.json()['text'])

if __name__=='__main__':
    # test_gpt()
    # test_glmmodel()
    test_api()
    # glm_generate()