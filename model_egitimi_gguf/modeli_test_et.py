from llama_cpp import Llama
from sabitler import model_path

llm = Llama(model_path=model_path)
def cikti_olustur(text):
    output = llm(
      f"Q: {text} A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
    )
    print(output['choices'][0]['text'])

while True:
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    if metin.lower() == 'exit':
        break
    cikti_olustur(metin.replace("I","ı").replace("İ","i").lower())