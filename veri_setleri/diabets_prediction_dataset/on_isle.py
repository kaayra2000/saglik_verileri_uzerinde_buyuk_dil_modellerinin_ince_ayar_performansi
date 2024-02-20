from itertools import product
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
    tum_kombinasyonlar = list(product(hipertansiyon_secenekleri, kalp_hastaligi_secenekleri,
                                       secenekler_bmi[bmi_hesapla(row["bmi"])], secenekler_yas[yas_getir(row["age"])], [cinsiyet_getir(row["gender"])],
                                       secenekler_sigara_kullanimi[sigara_kullanimi_cevir(row["smoking_history"])],
                                        secenekler_hba1c[hba1c_hesapla(row["HbA1c_level"])],secenekler_kan_sekeri[kan_sekeri_hesapla(row["blood_glucose_level"])],
                                       diyabet_durumu_secenekleri))
    tum_cumleler = []
    for kombinasyon in tum_kombinasyonlar:
        cumleler = [
            f'{kombinasyon[0]} ',
            f'{kombinasyon[1]} ',
            f'{kombinasyon[2]} ',
            f'{kombinasyon[3]} ',
            f'{kombinasyon[4]} ',
            f'{kombinasyon[5]} ',
            f'{kombinasyon[6]} ',
            f'{kombinasyon[7]} ',
            f'{kombinasyon[8]} ',

        ] 
        tum_cumleler.append(cumleler)
    return tum_cumleler
on_isleyici = OnIsle(veri_seti_adi, tanimlayici_cumle_olustur)

on_isleyici.isle_kaydet()