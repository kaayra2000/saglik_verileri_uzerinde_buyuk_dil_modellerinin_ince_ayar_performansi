import pandas as pd
import random
from degiskenler import *

"""
    Veri setini işler ve metin dönüşümü yapar.
"""
veri_seti_adi = "diabetes_prediction_dataset.csv"
# veri setini oku ve data'ya at
data = pd.read_csv(veri_seti_adi)

# DataFrame oluşturma
df = pd.DataFrame(data)


def yas_getir(age):
    if age < 18:
        durum = "çocuk"
    elif age < 65:
        durum = "yetişkin"
    else:
        durum = "yaşlı"

    return random.choice(secenekler_yas[durum])


def cinsiyet_getir(cinsiyet):
    if cinsiyet == "Male":
        return "erkektir"
    else:
        return "kadındır"


def bmi_hesapla(bmi):
    if bmi < 18.5:
        durum = "Düşük"
    elif 18.5 <= bmi < 25:
        durum = "Normal"
    elif 25 <= bmi < 30:
        durum = "Fazla Kilolu"
    else:
        durum = "Obez"

    return random.choice(secenekler_bmi[durum])


def hba1c_hesapla(hba1c_seviyesi):
    if hba1c_seviyesi < 5.7:
        durum = "Normal"
    elif 5.7 <= hba1c_seviyesi < 6.5:
        durum = "Prediyabet"
    else:
        durum = "Diyabet"

    return "hba1c verisine göre " + random.choice(secenekler_hba1c[durum])


def kan_sekeri_hesapla(kan_sekeri_seviyesi):
    if kan_sekeri_seviyesi < 70:
        durum = "Düşük"
    elif 70 <= kan_sekeri_seviyesi <= 99:
        durum = "Normal"
    elif 100 <= kan_sekeri_seviyesi <= 125:
        durum = "Yüksek Normal"
    else:
        durum = "Çok Yüksek"

    return random.choice(secenekler_kan_sekeri[durum])


def sigara_kullanimi_cevir(sigara_kullanimi):
    if sigara_kullanimi in secenekler_sigara_kullanimi:
        return random.choice(secenekler_sigara_kullanimi[sigara_kullanimi])
    else:
        return "Tanımsız"


def rastgele_cumle_sec(secenekler):
    return random.choice(secenekler)


# Metin dönüşüm fonksiyonu (diabetes durumu hariç)
def turkceye_cevir(row):
    hipertansiyon_secenekleri = (
        hiper_tansiyon_var if row["hypertension"] == 1 else hiper_tansiyon_yok
    )
    kalp_hastaligi_secenekleri = (
        kalp_hastaligi_var if row["heart_disease"] == 1 else kalp_hastaligi_yok
    )
    diyabet_durumu_secenekleri = diyabet_var if row["diabetes"] == 1 else diyabet_yok
    kalp_hastaligi_cumle = rastgele_cumle_sec(kalp_hastaligi_secenekleri)
    diyabet_durumu_cumle = rastgele_cumle_sec(diyabet_durumu_secenekleri)
    hiper_tansiyon_cumle = rastgele_cumle_sec(hipertansiyon_secenekleri)
    cumleler = [
        kalp_hastaligi_cumle,
        diyabet_durumu_cumle,
        hiper_tansiyon_cumle,
        yas_getir(row["age"]),
        cinsiyet_getir(row["gender"]),
        sigara_kullanimi_cevir(row["smoking_history"]),
        bmi_hesapla(row["bmi"]),
        hba1c_hesapla(row["HbA1c_level"]),
        kan_sekeri_hesapla(row["blood_glucose_level"]),
    ]
    # rastgele sıralama
    random.shuffle(cumleler)
    return " ".join(cumleler)


# Dönüştürülen metinleri yeni bir sütuna ekleme
df["text"] = df.apply(turkceye_cevir, axis=1)
# Sadece 'diabetes' ve 'text' sütunlarını içeren yeni bir DataFrame oluştur
selected_columns = df["text"]
# Dönüştürülen metinleri yeni bir dosyaya kaydetme
output_filename = "labelsiz_" + veri_seti_adi
selected_columns.to_csv(output_filename, index=False)
