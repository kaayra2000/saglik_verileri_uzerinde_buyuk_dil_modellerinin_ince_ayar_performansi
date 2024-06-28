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

veri_yolu = 'avaliyev/test.json'

# JSON dosyasını okuyun
with open(veri_yolu, 'r') as file:
    data = json.load(file)

# Her girdiyi çevirin ve sonuçları yeni bir listeye kaydedin
translated_data = []
for item in data:
    translated_item = {
        "instruction": item["instruction"],
        "input": translate_text(item["input"]),
        "output": translate_text(item["output"])
    }
    translated_data.append(translated_item)
sonuc_yolu = veri_yolu.replace('.json', '_translated.json')
# Çevirilmiş verileri yeni bir JSON dosyasına kaydedin
with open(sonuc_yolu, 'w') as file:
    json.dump(translated_data, file, ensure_ascii=False, indent=4)