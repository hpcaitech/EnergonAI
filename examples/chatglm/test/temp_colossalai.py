from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("/data2/share/opt-6.7b", torch_dtype=torch.float16).cuda()
param = next(model.parameters())
print(param.dtype)

modeltwo = AutoModelForCausalLM.from_pretrained("/data2/share/opt-6.7b").cuda()
param = next(model.parameters())
print(param.dtype)

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("/data2/share/opt-6.7b", use_fast=False)

prompt = "Hello, I'm am conscious and"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

generated_ids = model.generate(input_ids)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
