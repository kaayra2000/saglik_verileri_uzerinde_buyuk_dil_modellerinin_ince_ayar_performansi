import json
import os
import sys
import re

sys.path.append("../..")
from fonksiyonlar import veri_yolu_al
from openai import OpenAI

# API anahtarını oku
with open("../../../api_key.txt", "r", encoding="utf-8") as file:
    api_key = file.read().strip()

# OpenAI istemcisini oluştur
client = OpenAI(api_key=api_key)

# batch_info.txt dosyasını oku
with open("batch_info.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Sonuçların kaydedileceği klasör
output_folder = "sonuclar"
output_prefix = veri_yolu_al()
os.makedirs(output_folder, exist_ok=True)

# Batch ID'lerini içeren satırları işle
for line in lines:
    line = line.strip()
    if line:
        # output_file_id'yi al
        output_file_id_match = re.search(r"output_file_id='(file-[^']+)'", line)
        if output_file_id_match:
            output_file_id = output_file_id_match.group(1)

            # Batch içeriğini al
            response = client.files.content(output_file_id)
            content = response.content.decode("utf-8")

            # İçeriği JSON objelerine ayır
            data = {"data": []}
            for line in content.splitlines():
                json_obj = json.loads(line)
                data["data"].append(json_obj)

            # İlk ve son elemanın custom_id'sini al
            first_custom_id = data["data"][0]["custom_id"]
            last_custom_id = data["data"][-1]["custom_id"]

            # Dosya adını oluştur
            start_id = first_custom_id.split("-")[1]
            end_id = last_custom_id.split("-")[1]
            output_postfix = f"{start_id}-{end_id}_output.json"
            output_file_name = f"{output_prefix}_{output_postfix}"

            # Sonuçları dosyaya yaz
            with open(
                os.path.join(output_folder, output_file_name), "w", encoding="utf-8"
            ) as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)

            print(f"{output_file_name} dosyası oluşturuldu.")

print("Tüm batch işlemleri tamamlandı.")
