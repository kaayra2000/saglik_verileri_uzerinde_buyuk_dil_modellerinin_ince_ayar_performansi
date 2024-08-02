import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="CSV dosyalarını yükle ve görüntüle")
parser.add_argument("test_file", type=str, help="Test CSV dosyasının yolu")
parser.add_argument(
    "chat_doctor_file", type=str, help="Chat-Doktor CSV dosyasının yolu"
)

args = parser.parse_args()

# Dosyaların yüklenmesi
test_df = pd.read_csv(args.test_file)
chat_doctor_file = pd.read_csv(args.chat_doctor_file)

# 1. Test ve Doktor-Mistral dosyalarındaki 'question' kolonlarını karşılaştırarak farklı olan indeksleri tespit et
farkli_indisler = []

for i, (test_question, doktor_question) in enumerate(
    zip(test_df["question"], chat_doctor_file["question"])
):
    if test_question != doktor_question:
        farkli_indisler.append(i)

# Farklı olan indeksleri ekrana yazdır
print(
    "Test ve Doktor-Mistral dosyalarındaki farklı 'question' kolonuna sahip indeksler:"
)
print(farkli_indisler)

# 2. Chat-Doktor dosyasındaki tekrarlanan indeksleri ve aynı soruya sahip olanları bul
tekrarlanan_indisler = chat_doctor_file[
    chat_doctor_file.duplicated(subset=["index"], keep=False)
]
print(
    f"Chat-Doktor dosyasındaki tekrarlanan indeksler: {tekrarlanan_indisler['index'].tolist()}"
)
