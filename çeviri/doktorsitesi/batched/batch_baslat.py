import time
from openai import OpenAI


def get_file_id(file_line):
    # FileObject(id='file-EWOCP7WaRmJVUmdJhibV4tcf', bytes=1570909, created_at=1720417181, filename='sohbetler20000_batch_1310-1809.jsonl', object='file', purpose='batch', status='processed', status_details=None)
    parts = file_line.split(",")
    for part in parts:
        if "id=" in part:
            return part.split("'")[1]
    return None


# OpenAI API anahtarını oku
with open("../../../api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

# batch_file_info.txt dosyasını oku
with open("batch_file_info.txt", "r") as file:
    file_lines = file.readlines()

for line in file_lines:
    batch_input_file_id = get_file_id(line.strip())
    if batch_input_file_id:
        # Yeni batch başlat
        sonuc = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "20k doktorsitesi"},
        )

        batch_id = sonuc.id
        print(f"Batch {batch_id} başlatıldı.")
        # Batch durumunu kontrol et
        while True:
            sonuc = client.batches.retrieve(batch_id=batch_id)
            status = sonuc.status

            if status == "completed":
                print(f"Batch {batch_id} başarıyla tamamlandı.")
                break
            elif status == "failed":
                print(f"Batch {batch_id} başarısız oldu.")
                exit(1)  # İşlemi sonlandır
            elif status == "in_progress":
                print(f"Batch {batch_id} devam ediyor.")
            elif status == "finalizing":
                print(f"Batch {batch_id} tamamlanıyor.")
            elif status == "validating":
                print(f"Batch {batch_id} doğrulanıyor.")
            elif status == "cancelling":
                print(f"Batch {batch_id} iptal ediliyor.")
            elif status == "cancelled":
                print(f"Batch {batch_id} iptal edildi.")
                exit(1)  # İşlemi sonlandır
            time.sleep(60)  # 60 saniye bekle ve tekrar kontrol et

        # Batch bilgilerini yaz
        with open("batch_info.txt", "a") as file:
            file.write(str(sonuc) + "\n")

print("Tüm batch işlemleri tamamlandı.")
