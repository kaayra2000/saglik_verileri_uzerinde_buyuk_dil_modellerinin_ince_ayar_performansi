from openai import OpenAI
import json


with open('../api_key.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

def translate_text(text, source_language="en", target_language="tr"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
         messages=[
            {"role": "system", "content": f"You are a helpful assistant that translates {source_language} to {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

veri_yolu = 'avaliyev/test_translated.json'

# JSON dosyasını okuyun
with open(veri_yolu, 'r') as file:
    data = json.load(file)

sonuc_yolu = veri_yolu.replace('.json', '_tr.json')
tr_instruction = "Eğer bir doktor iseniz, lütfen hastanın tarifine dayanarak tıbbi soruları cevaplayın."
# Dosyayı yazma modunda aç ve başlat
with open(sonuc_yolu, 'w') as file:
    file.write('[\n')  # JSON dosyasının başını yaz

    for i, item in enumerate(data):
        translated_item = {
            "instruction": tr_instruction,
            "input": translate_text(["input"]),
            "output": translate_text(item["output"]),
        }
        
        # Öğeyi JSON formatına dönüştür
        json_item = json.dumps(translated_item, ensure_ascii=False, indent=4)
        
        # Dosyaya yaz
        if i > 0:
            file.write(',\n')  # Önceki öğelerden sonra virgül ekle
        file.write(json_item)
    
    file.write('\n]')  # JSON dosyasının sonunu yaz