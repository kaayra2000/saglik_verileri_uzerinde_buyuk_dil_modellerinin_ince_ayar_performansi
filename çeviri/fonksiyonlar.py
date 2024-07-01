import json
from openai import OpenAI
from deep_translator import GoogleTranslator
import argparse


def translate_text_google(text: str, translator: GoogleTranslator):
    translated_text = translator.translate(text)
    return translated_text


def veri_yolu_al():
    # Komut satırı argümanlarını alma
    parser = argparse.ArgumentParser(description="Translate text from a given file.")
    parser.add_argument(
        "veri_yolu", type=str, help="Çevirilecek verinin bulunduğu dosya yolu"
    )
    args = parser.parse_args()

    # Dosya yolunu kullanarak işlemleri yapma
    veri_yolu = args.veri_yolu
    return veri_yolu


def translate_text_openai(
    text, client: OpenAI, source_language="en", target_language="tr"
):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": f"Translate the following {source_language} text to {target_language} without changing the style or adding additional information.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def cevir_kaydet(veri_yolu: str, translate_text: callable, client: any):
    # JSON dosyasını okuyun
    with open(veri_yolu, "r") as file:
        data = json.load(file)

    sonuc_yolu = veri_yolu.replace(".json", "_translated.json")
    tr_instruction = "Eğer bir doktor iseniz, lütfen hastanın tarifine dayanarak tıbbi soruları cevaplayın."
    # Dosyayı yazma modunda aç ve başlat
    with open(sonuc_yolu, "w") as file:
        file.write("[\n")

    for i, item in enumerate(data):
        translated_item = {
            "instruction": tr_instruction,
            "input": translate_text(item["input"], client),
            "output": translate_text(item["output"], client),
        }

        # Öğeyi JSON formatına dönüştür
        json_item = json.dumps(translated_item, ensure_ascii=False, indent=4)

        # Dosyaya yaz
        with open(sonuc_yolu, "a") as file:
            if i > 0:
                file.write(",\n")  # Önceki öğelerden sonra virgül ekle
            file.write(json_item)

    # JSON dosyasının sonunu yaz
    with open(sonuc_yolu, "a") as file:
        file.write("\n]")
