from transformers import TFGPT2LMHeadModel, AutoTokenizer
from sabitler import *
model_path_without_label = os.path.join(model_path_without_label, model_adi)
model = TFGPT2LMHeadModel.from_pretrained(model_path_without_label)

tokenizer = AutoTokenizer.from_pretrained(model_path_without_label)

def cikti_olustur(text):
    inputs = tokenizer(text, return_tensors="tf")  # Metni tokenleştirin
    output = model.generate(inputs["input_ids"])  # Modelden metin üretin
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Tokenları metne geri dönüştürün
    print(generated_text)

while True:
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    if metin.lower() == 'exit':
        break
    cikti_olustur(metin.replace("I","ı").replace("İ","i").lower())

