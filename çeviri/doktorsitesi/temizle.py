import csv
import sys
from openai import OpenAI
import concurrent.futures
import os

sys.path.append("..")
from fonksiyonlar import veri_yolu_al

with open("../../api_key.txt", "r") as file:
    api_key = file.read().strip()


def clean_openai(
    text,
    client: OpenAI,
    model="ft:gpt-3.5-turbo-0125:kayra::9i7HydFK",
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Kullanıcının yazdığı metni temizle ve gereksiz bilgileri kaldır. Sadece gerekli olan ve anlamlı bilgileri bırak. Bu süreçte aşağıdaki kurallara uy:\nAdres cümlelerini, anlamı bozmadan direkt çıkart. Adrese atıf yapma.\nLinkleri doğrudan çıkart, anlamı bozmadan linklere atıf yapma.\nÜnvan ve isimleri doğrudan çıkart. Örneğin, 'Prof. Dr. Ahmet' yerine 'Dr.' ya da 'doktor' yeterli.\nYanlış yazılan kelimeleri düzelt. Örneğin, 'Yadrım edin' yerine 'Yardım edin'.\nTarihleri, anlam bütünlüğünü bozmayacak şekilde tamamen kaldır.\nBağlam bağımlı cümleleri, anlam bütünlüğünü bozmayacak şekilde tamamen kaldır. Örneğin, 'geçen gün sizinle görüşmüştük' gibi cümleleri çıkar.\nBu bir doktor-hasta verisi olduğu için, soru cevabı içeren veri seti temizlemesinde kullanılacak ve sonrasında bir chatbot eğitmede kullanılacak. Bu bilgiye göre temizleme yap.",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        n=1,
    )
    return response.choices[0].message.content.strip()


def process_row(row, client, model):
    row["question_content"] = clean_openai(row["question_content"], client, model)
    row["question_answer"] = clean_openai(row["question_answer"], client, model)
    return row


def process_and_write_rows(
    data,
    start_index,
    hedef_yol,
    client,
    model="ft:gpt-3.5-turbo-0125:kayra::9i7HydFK",
    chunk_size=50,
):
    total_rows = len(data)
    for chunk_start in range(start_index, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_data = data[chunk_start:chunk_end]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(process_row, row, client, model): i
                for i, row in enumerate(chunk_data, start=chunk_start)
            }

            results = [None] * len(chunk_data)

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results[index - chunk_start] = future.result()

        with open(hedef_yol, "a", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
            for row in results:
                if row is not None:
                    writer.writerow(row)


def cevir_kaydet_csv(veri_yolu, translator):
    hedef_yol = veri_yolu.replace(".csv", "_cleaned.csv")

    # Girdi dosyasını oku
    with open(veri_yolu, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = [field for field in reader.fieldnames if field != "Title"]
        data = list(reader)

    # Mevcut çevrilmiş dosyanın var olup olmadığını kontrol edin
    start_index = 0
    if os.path.exists(hedef_yol):
        with open(hedef_yol, "r", newline="", encoding="utf-8") as outfile:
            reader = csv.DictReader(outfile)
            existing_data = list(reader)
            start_index = len(existing_data)

    # Çıktı dosyasını oluştur ve başlık satırını yaz (eğer yeni dosya ise)
    if start_index == 0:
        with open(hedef_yol, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

    process_and_write_rows(data, start_index, hedef_yol, translator)


veri_yolu = veri_yolu_al()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)
cevir_kaydet_csv(veri_yolu, client)
