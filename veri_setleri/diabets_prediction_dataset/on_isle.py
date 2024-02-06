import random
import sys
from degiskenler import *
import os
from fonksiyonlar import *
# Mevcut dosyanın bulunduğu dizinin tam yolunu al
mevcut_dizin = os.path.dirname(os.path.abspath(__file__))

# Bir üst dizinin yolunu bul
ust_dizin = os.path.abspath(os.path.join(mevcut_dizin, os.pardir))

# Bu yolu sys.path'e ekle
sys.path.append(ust_dizin)
from yuksek_dusuk import *
"""
    Veri setini işler ve metin dönüşümü yapar.
"""
veri_seti_adi = "diabetes_prediction_dataset.csv"
# veri setini oku ve data'ya at

# Metin dönüşüm fonksiyonu (diabetes durumu hariç)
def tanimlayici_cumle_olustur(row):
    hipertansiyon_secenekleri = (
        hiper_tansiyon_var if row["hypertension"] == 1 else hiper_tansiyon_yok
    )
    kalp_hastaligi_secenekleri = (
        kalp_hastaligi_var if row["heart_disease"] == 1 else kalp_hastaligi_yok
    )
    diyabet_durumu_secenekleri = diyabet_var if row["diabetes"] == 1 else diyabet_yok
    kalp_hastaligi_cumle = rastgele_cumle_sec(kalp_hastaligi_secenekleri)
    diyabet_durumu_cumle = CEVAP + rastgele_cumle_sec(diyabet_durumu_secenekleri)
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
    return cumleler
on_isleyici = OnIsle(veri_seti_adi, tanimlayici_cumle_olustur)

on_isleyici.isle_kaydet()