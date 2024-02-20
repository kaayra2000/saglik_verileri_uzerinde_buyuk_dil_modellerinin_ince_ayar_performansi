import pandas as pd
import os
import sys
from degiskenler import *
from fonksiyonlar import *
from itertools import product
# Mevcut dosyanın bulunduğu dizinin tam yolunu al
mevcut_dizin = os.path.dirname(os.path.abspath(__file__))

# Bir üst dizinin yolunu bul
ust_dizin = os.path.abspath(os.path.join(mevcut_dizin, os.pardir))

# Bu yolu sys.path'e ekle
sys.path.append(ust_dizin)
from yuksek_dusuk import *
# Veri setini yükle
dataset_path = "Breast_cancer_data.csv"


# Tanımlayıcı cümleler oluşturma fonksiyonu
def tanimlayici_cumle_olustur(row):
    # Tüm mümkün kombinasyonları oluşturmak
    tum_kombinasyonlar = list(product(yaricap_secenekleri, doku_secenekleri, cevre_secenekleri, alan_secenekleri, duzgunluk_secenekleri, tani_secenekleri[row["diagnosis"]]))
    tum_cumleler = []
    for kombinasyon in tum_kombinasyonlar:
        cumleler = [
            f'{kombinasyon[0].format(radius_kategori_belirle(row["mean_radius"]))} ',
            f'{kombinasyon[1].format(texture_kategori_belirle(row["mean_texture"]))} ',
            f'{kombinasyon[2].format(perimeter_kategori_belirle(row["mean_perimeter"]))} ',
            f'{kombinasyon[3].format(area_kategori_belirle(row["mean_area"]))} ',
            f'{kombinasyon[4].format(smoothness_kategori_belirle(row["mean_smoothness"]))} ',
            f'{kombinasyon[5]} '
        ] 
        tum_cumleler.append(cumleler)
    return tum_cumleler

on_isleyici = OnIsle(dataset_path, tanimlayici_cumle_olustur)
on_isleyici.isle_kaydet()
