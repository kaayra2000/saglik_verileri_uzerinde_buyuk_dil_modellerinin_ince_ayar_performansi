import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import preprocess_turkish_text
import os
from sabitler import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hatalar hariç tüm logları

os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)
tokenizer = AutoTokenizer.from_pretrained(model_adi)
model = AutoModelForCausalLM.from_pretrained(model_adi)

# Cihazı belirle (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt_text, max_length=100):
    # Girdi metnini tokenize et ve attention mask oluştur
    encoded_input = tokenizer.encode(prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    
    # Metin üretimi
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length + 20,  # Cevap için ekstra uzunluk
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )
    
    # Üretilen çıktıyı metne dönüştür
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text.replace(prompt_text,"")

while True:
    # Kullanıcıdan soru girdisi alma
    prompt_text = input("Soru: ")

    # Tam soru metnini oluştur
    full_prompt_text = f"Soru: {prompt_text} Cevap:"

    # Metin üretimi ve çıktının gösterilmesi
    generated_text = generate_text(full_prompt_text)
    print("Modelin ürettiği cevap:", generated_text)
