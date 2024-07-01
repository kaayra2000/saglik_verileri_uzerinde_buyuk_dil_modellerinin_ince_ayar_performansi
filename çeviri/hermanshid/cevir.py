import csv
import os, sys
from deep_translator import GoogleTranslator

sys.path.append("..")
from fonksiyonlar import translate_text_google, veri_yolu_al


def cevir_kaydet_csv(veri_yolu, translator):
    hedef_yol = veri_yolu.replace(".csv", "_translated.csv")

    # Girdi dosyasını oku
    with open(veri_yolu, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
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

    # Satırları teker teker işle ve yaz
    for i, row in enumerate(data[start_index:], start=start_index):
        row["question"] = translate_text_google(row["question"], translator)
        row["answer"] = translate_text_google(row["answer"], translator)
        # Çevrilen satırı yazmak için dosyayı aç, yaz ve kapat
        with open(hedef_yol, "a", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writerow(row)


veri_yolu = veri_yolu_al()
translator = GoogleTranslator(source="id", target="tr")
# Örnek kullanım
cevir_kaydet_csv(veri_yolu, translator)
