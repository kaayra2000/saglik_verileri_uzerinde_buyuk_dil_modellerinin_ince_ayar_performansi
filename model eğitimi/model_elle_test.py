from transformers import pipeline
from sabitler import model_name
generator = pipeline('text-generation', model=model_name)


# while true ile sürekli girdi al
while True:
    # Girdi al
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    # Çıkış koşulu
    if metin.lower() == 'exit':
        break
    # Girdiyi modelden geçir
    print(generator(metin, max_length=20, num_return_sequences=1))