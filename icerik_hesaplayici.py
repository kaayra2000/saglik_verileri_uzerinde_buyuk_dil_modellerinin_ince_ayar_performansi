import csv
import json

from veri_setleri.degiskenler import diyabet_var, diyabet_yok, ANA_VERI_SETI

# Sayıcılar
diyabet_var_count = 0
diyabet_yok_count = 0

# CSV dosyasını satır satır okuyacak
try:
    with open(ANA_VERI_SETI, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_text = ' '.join(row)  # Satırı metin olarak birleştir
            if any(phrase in row_text for phrase in diyabet_yok):
                diyabet_yok_count += 1
            elif any(phrase in row_text for phrase in diyabet_var):
                diyabet_var_count += 1

    # Toplam sayı
    total = diyabet_var_count + diyabet_yok_count

    # Sonuçları JSON olarak kaydet
    results = {
        "toplam": total,
        "diyabet_var": diyabet_var_count,
        "diyabet_yok": diyabet_yok_count
    }

    with open('ana_veri_seti_icerik.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=4)

except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")