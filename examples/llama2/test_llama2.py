from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model_path='/data2/share/LLaMa2/7b-chat-hf'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()

state_dict=model.state_dict()

# 获取两层的 rotary_emb.inv_freq 权重
rotary_emb_inv_freq_0 = state_dict["model.layers.0.self_attn.rotary_emb.inv_freq"]
rotary_emb_inv_freq_1 = state_dict["model.layers.1.self_attn.rotary_emb.inv_freq"]
rotary_emb_inv_freq_31 = state_dict["model.layers.31.self_attn.rotary_emb.inv_freq"]

# 比较两个权重是否相等
are_equal = torch.all(torch.eq(rotary_emb_inv_freq_0,rotary_emb_inv_freq_1))
print("Are the rotary_emb.inv_freq weights equal? ", are_equal)
are_equal = torch.all(torch.eq(rotary_emb_inv_freq_0,rotary_emb_inv_freq_31))
# are_equal = torch.all(torch.eq(rotary_emb_inv_freq_0, rotary_emb_inv_freq_1))
print("Are the rotary_emb.inv_freq weights equal? ", are_equal)

# for name,weight in state_dict.items():
    # print(f'Name:{name},shape:{weight.shape}')
    # print(f'')
    # print("\n")

# for name , parm in model.named_parameters():
#     print(name,';',parm.size())

# model.to(torch.device)
model=model.eval()
# input_text='今天天气大概 25度，有点小雨，吹着风，我想去户外散步，应该穿什么样的衣服裤子鞋子搭'
input_text='please say something about Asia'
from tqdm import tqdm

input_text=tokenizer.encode(input_text,return_tensors ='pt')
input_text=input_text.to(model.device)

for i in tqdm(range(1)):
# while True:
    # input_text=input('请输入内容:\n')
    # input_text=tokenizer.encode(input_text,return_tensors ='pt')
    # input_text=input_text.to(model.device)
    output=model.generate(input_text , max_length = 200)
    output=tokenizer.decode(output[0])
    print(output)

print('down')
