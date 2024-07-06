from openai import OpenAI
import json
# API anahtarını oku
with open("../../../api_key.txt", "r", encoding="utf-8") as file:
    api_key = file.read().strip()

# OpenAI istemcisini oluştur
client = OpenAI(api_key=api_key)

# Dosya içeriğini al
file_id = "file-uXh7ifyBvmEI41KBDinv9Klg"
response = client.files.content(file_id)

# İçeriği UTF-8 olarak dekode et ve satır satır işle
content = response.content.decode("utf-8")
data = {}
data["data"] = []
for line in content.splitlines():
        json_obj = json.loads(line)
        data["data"].append(json_obj)


# JSONL içeriğini UTF-8 olarak kaydet
with open("output.json", "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
