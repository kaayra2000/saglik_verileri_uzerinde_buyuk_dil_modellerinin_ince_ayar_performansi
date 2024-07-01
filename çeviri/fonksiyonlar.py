import json
from openai import OpenAI
from deep_translator import GoogleTranslator
import argparse
import os


def translate_text_google(
    text: str, translator: GoogleTranslator, max_length: int = 5000
):
    def split_text(text, max_length):
        words = text.split(" ")
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    chunks = split_text(text, max_length)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return "\n".join(translated_chunks)


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


def remove_trailing_bracket(file_path: str):
    try:
        with open(file_path, "r+") as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell()

            # Geriye doğru git ve son `}` karakterine kadar olan gereksiz karakterleri sil
            while pos > 0:
                pos -= 1
                file.seek(pos, os.SEEK_SET)
                char = file.read(1)
                if char == "}":
                    break
            # `}` karakterinden sonrasını sil
            file.truncate(pos + 1)
        print(
            f"{file_path} dosyasının sonundaki gereksiz karakterler başarıyla silindi."
        )
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


def cevir_kaydet_json(veri_yolu: str, translate_text: callable, client: any):
    # JSON dosyasını okuyun
    with open(veri_yolu, "r") as file:
        data = json.load(file)

    sonuc_yolu = veri_yolu.replace(".json", "_translated.json")
    tr_instruction = "Eğer bir doktor iseniz, lütfen hastanın tarifine dayanarak tıbbi soruları cevaplayın."

    # Mevcut çevrilmiş dosyanın var olup olmadığını kontrol edin
    start_index = 0
    if os.path.exists(sonuc_yolu):
        with open(sonuc_yolu, "r+") as file:
            existing_data = json.load(file)
            start_index = len(existing_data)
        remove_trailing_bracket(sonuc_yolu)
    else:
        # Dosyayı yazma modunda aç ve başlat
        with open(sonuc_yolu, "w") as file:
            file.write("[\n")

    try:
        # Çeviri işlemini kaldığı yerden devam ettirin
        for i, item in enumerate(data[start_index:], start=start_index):
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
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        # JSON dosyasının sonunu yaz
        with open(sonuc_yolu, "a") as file:
            file.write("\n]")
