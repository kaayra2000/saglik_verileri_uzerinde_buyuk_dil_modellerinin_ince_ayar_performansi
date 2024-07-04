from deep_translator import GoogleTranslator
import sys
import json
import concurrent.futures
import os

sys.path.append("..")
from fonksiyonlar import translate_text_google, veri_yolu_al, remove_trailing_bracket

translator = GoogleTranslator(source="en", target="tr")


def process_item(item, tr_instruction, client):
    input = translate_text_google(item["input"], client)
    output = translate_text_google(item["output"], client)
    if output is None or input is None:
        raise Exception("Çeviri yapılamadı.")
    translated_item = {
        "instruction": tr_instruction,
        "input": input,
        "output": output,
    }
    return translated_item


def cevir_kaydet_json(veri_yolu: str, client: GoogleTranslator):
    with open(veri_yolu, "r") as file:
        data = json.load(file)

    sonuc_yolu = veri_yolu.replace(".json", "_translated.json")
    tr_instruction = "Eğer bir doktor iseniz, lütfen hastanın tarifine dayanarak tıbbi soruları cevaplayın."

    start_index = 0
    if os.path.exists(sonuc_yolu):
        with open(sonuc_yolu, "r+") as file:
            existing_data = json.load(file)
            start_index = len(existing_data)
        remove_trailing_bracket(sonuc_yolu)
    else:
        with open(sonuc_yolu, "w") as file:
            file.write("[\n")

    try:
        chunk_size = 50
        total_items = len(data)
        for chunk_start in range(start_index, total_items, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_items)
            chunk_data = data[chunk_start:chunk_end]

            results = [None] * len(chunk_data)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_index = {
                    executor.submit(process_item, item, tr_instruction, client): idx
                    for idx, item in enumerate(chunk_data)
                }

                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()

            with open(sonuc_yolu, "a") as file:
                for i, result in enumerate(results):
                    if chunk_start + i > 0 or start_index > 0:
                        file.write(",\n")
                    json_item = json.dumps(result, ensure_ascii=False, indent=4)
                    file.write(json_item)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        with open(sonuc_yolu, "a") as file:
            file.write("\n]")


veri_yolu = veri_yolu_al()
cevir_kaydet_json(veri_yolu, translator)
