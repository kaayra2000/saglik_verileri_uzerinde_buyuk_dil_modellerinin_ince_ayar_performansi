from openai import OpenAI
import json
import time

with open("../../api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

with open("fine_tune_data.json", "r") as file:
    data = json.load(file)

# JSONL formatında kaydetme
with open("output_chat.jsonl", "w", encoding="utf-8") as f:
    for entry in data["data"]:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + "\n")

result = client.files.create(
    file=open("output_chat.jsonl", "rb"),
    purpose="fine-tune",
)
print(f"Dodya id {result.id} olarak yüklendi.")

result = client.fine_tuning.jobs.create(
    training_file=result.id,
    model="gpt-3.5-turbo-0125",
)
print(f"Eğitim başladı. Eğitim id: {result.id}")
while True:
    if result.status == "succeeded":
        print("Eğitim başarıyla tamamlandı.")
        break
    elif result.status == "failed":
        print("Eğitim başarısız oldu.")
        break
    elif result.status == "running":
        print("Eğitim devam ediyor...")
    result = client.fine_tuning.jobs.retrieve(result.id)
    time.sleep(3)
