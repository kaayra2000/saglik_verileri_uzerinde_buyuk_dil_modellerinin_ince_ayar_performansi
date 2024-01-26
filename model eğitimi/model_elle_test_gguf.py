from ctransformers import AutoModelForCausalLM
from sabitler import model_name, model_file
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_file, model_type="mistral", gpu_layers=0)
while True:
    # Girdi al
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    # Çıkış koşulu
    if metin.lower() == 'exit':
        break
    # Girdiyi modelden geçir
    print(llm(metin))
