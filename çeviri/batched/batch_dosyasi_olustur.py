import csv
import sys
import os
import json

sys.path.append("..")
from fonksiyonlar import veri_yolu_al


def create_batch_request(content, index, type):
    return {
        "custom_id": f"request-{index}-{type}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "ft:gpt-3.5-turbo-0125:kayra::9ixgAbko",
            "messages": [
                {
                    "role": "system",
                    "content": "İngilizceden Türkçe'ye çeviri yap.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "max_tokens": 4096,
        },
    }


def save_batch_requests(data, batch_file_path, start_index):
    with open(batch_file_path, "w", encoding="utf-8") as batch_file:
        for index, row in enumerate(data[start_index:], start=start_index):
            request = create_batch_request(row["Question"], index, "question")
            batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            request = create_batch_request(row["Answer"], index, "answer")
            batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            request = create_batch_request(row["Title"], index, "title")
            batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")


def cevir_kaydet_jsonl(veri_yolu):
    batch_file_path = veri_yolu.replace(".csv", "_batch.jsonl")
    cleaned_file_path = veri_yolu.replace(".csv", "_cleaned.csv")
    # Girdi dosyasını oku
    with open(veri_yolu, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader((line.replace("\0", "") for line in infile))
        data = list(reader)

    # Mevcut çevrilmiş dosyanın var olup olmadığını kontrol edin
    start_index = 0
    if os.path.exists(cleaned_file_path):
        with open(cleaned_file_path, "r", newline="", encoding="utf-8") as outfile:
            reader = csv.DictReader(outfile)
            existing_data = list(reader)
            start_index = len(existing_data)

    save_batch_requests(data, batch_file_path, start_index)


veri_yolu = veri_yolu_al()
cevir_kaydet_jsonl(veri_yolu)
