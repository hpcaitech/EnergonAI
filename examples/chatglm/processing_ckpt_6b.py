import os
import torch
from multiprocessing import Pool
import pdb

# download pytorch model ckpt in https://huggingface.co/facebook/opt-66b/tree/main
# you can use whether wget or git lfs
print('step1')
path = "/data/share/chatglm-6b/only_model"
new_path = "/data2/zxy/4_energon_glm"
# 使用显卡的个数
graphics_card=4 

assert os.path.isdir(path)
files = []
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath):
        files.append(filepath)
print('step2')
import os

# ...
# Ensure all files exist and are not empty
for file in files:
    if not os.path.exists(file):
        print(f"File does not exist: {file}")
    elif os.path.getsize(file) == 0:
        print(f"File is empty: {file}")
    else:
        print(f'{file} is correct')
# ...


with Pool(graphics_card) as pool:
    ckpts = pool.map(torch.load, files)
print('step3')
restored = {}
for ckpt in ckpts:
    for k,v in ckpt.items():
        # if(k[0] == 'm'):
        #     k = k[6:]      
        if(k == "lm_head.weight"):
            k = "head.dense.weight"
        if(k == "transformer.final_layernorm.weight"):
            k = "decoder.layer_norm.weight"
        if(k == "transformer.final_layernorm.bias"):
            k = "decoder.layer_norm.bias"
        restored[k] = v
restored["decoder.version"] = "0.0"

print('step4')
# pdb.set_trace()
split_num = len(restored.keys()) // graphics_card
# print(split_num)
count = 0
file_count = 1
tmp = {}
for k,v in restored.items():
    print(k)
    tmp[k] = v
    count = count + 1    
    if(count == split_num):
        print(count)
        filename = str(file_count) + "-restored.pt"
        torch.save(tmp, os.path.join(new_path, filename))
        file_count = file_count + 1
        count = 0
        tmp = {}
print('step5')# this is last other weight 
filename = str(file_count) + "-restored.pt"
torch.save(tmp, os.path.join(new_path, filename))