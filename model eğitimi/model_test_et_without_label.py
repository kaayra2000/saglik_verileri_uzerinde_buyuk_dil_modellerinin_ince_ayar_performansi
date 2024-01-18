import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sabitler import *
os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             revision="main",
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def cikti_olustur(text):
    text = f'''
        ### Instruction:  {text} ### Response:
        '''
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output = model.generate(inputs=input_ids,max_new_tokens=512,pad_token_id=tokenizer.eos_token_id,top_k=50, do_sample=True,
            top_p=0.95)
    response = tokenizer.decode(output[0])

    print(response)

while True:
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    if metin.lower() == 'exit':
        break
    cikti_olustur(metin.replace("I","ı").replace("İ","i").lower())

