import os
from sabitler import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import preprocess_turkish_text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hatalar hariç tüm logları

os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)
tokenizer = AutoTokenizer.from_pretrained(model_adi)
model = AutoModelForCausalLM.from_pretrained(model_adi)

# Cihazı belirle (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt_text, max_length=500):
    prompt_text = preprocess_turkish_text(prompt_text)
    # Girdi metnini tokenize et ve attention mask oluştur
    encoded_input = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded_input = encoded_input.to(device)

    # Metin üretimi
    output_sequences = model.generate(
        input_ids=encoded_input["input_ids"],
        attention_mask=encoded_input["attention_mask"],
        max_length=max_length + len(encoded_input["input_ids"][0]),
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # EOS token ID'sini pad_token_id olarak kullan
    )
    
    # Üretilen çıktıyı metne dönüştür
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    
    # Girdi metnini çıktı metinden ayır
    text = text[len(tokenizer.decode(encoded_input["input_ids"][0], clean_up_tokenization_spaces=True)):]
    
    return text.strip()
while True:

    # Kullanıcıdan metin girdisi alma
    prompt_text = input("Lütfen bir metin girin: ")

    # Metin üretimi ve çıktının gösterilmesi
    generated_text = generate_text(prompt_text)
    print("Modelin ürettiği metin:", generated_text)