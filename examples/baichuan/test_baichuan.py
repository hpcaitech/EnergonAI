from transformers import AutoModelForCausalLM, AutoTokenizer

path='/data2/share/Baichuan-7B'
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
state_dict=model.state_dict()

for name,weight in state_dict.items():
    print(f'{name},{weight.shape}')


inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))



