import csv
import requests
import io

# CSV dosyasını UTF-8 formatında indirme ve okuma
link = "https://docs.google.com/spreadsheets/d/1T-1Q0pecI0NePttA8o4OuqUjxvqdZ7-ty4aK3-eF-bw/pub?output=csv"
response = requests.get(link)
response.encoding = "utf-8"
csv_data = io.StringIO(response.text)
reader = csv.reader(csv_data)

# Başlık satırını okuma ve model isimlerini belirleme
header = next(reader)
model_names = ["trendyol_mistral", "sambalingo_llama2", "meta_llama3", "cosmos_llama3"]

# İndis listesi
indices = [
    27012,
    7057,
    3190,
    4432,
    3175,
    18466,
    31196,
    10410,
    24456,
    32027,
    31918,
    4858,
    22412,
    30396,
    25067,
    27027,
    7307,
    20361,
    17740,
    5006,
]

# Her satır için işlem yapma
for row in reader:
    # Kişi bilgilerini alma
    name = row[2]

    # Boş satırları atla
    if not name:
        continue

    title = row[5]
    expertise = row[4]
    institution = row[3]
    student_class = row[6]  # Sınıf bilgisi

    # Eğer öğrenciyse, unvan kısmına sınıf bilgisini ekle
    if title.lower() == "öğrenci":
        title = student_class

    # Yeni CSV dosyası için veri hazırlama
    new_data = []
    question_number = 1
    index_counter = 0

    # Her 4 sütun için bir soru oluşturma (7. sütundan başlayarak)
    for i in range(7, len(row), 4):
        current_index = indices[index_counter] if index_counter < len(indices) else 0
        for j, model in enumerate(model_names):
            if i + j < len(row):
                new_data.append(
                    [
                        name,
                        title,
                        question_number,
                        current_index,
                        model,
                        expertise,
                        institution,
                        row[i + j],
                    ]
                )
        question_number += 1
        index_counter += 1

    # Yeni CSV dosyasını UTF-8 formatında oluşturma ve yazma
    filename = f"{name.replace(' ', '_')}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["isim", "unvan", "soru", "indis", "model_adi", "uzmanlık", "kurum", "puan"]
        )
        writer.writerows(new_data)

print("İşlem tamamlandı. CSV dosyaları UTF-8 formatında oluşturuldu.")
