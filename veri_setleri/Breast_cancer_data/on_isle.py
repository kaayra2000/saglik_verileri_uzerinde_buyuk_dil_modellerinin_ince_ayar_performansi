import pandas as pd
import os
import sys
from degiskenler import *
from fonksiyonlar import *
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
    cumleler = []
    cumleler.append(rastgele_cumle_sec(yaricap_secenekleri).format(radius_kategori_belirle(row['mean_radius'])))
    cumleler.append(rastgele_cumle_sec(doku_secenekleri).format(texture_kategori_belirle(row['mean_texture'])))
    cumleler.append(rastgele_cumle_sec(cevre_secenekleri).format(perimeter_kategori_belirle(row['mean_perimeter'])))
    cumleler.append(rastgele_cumle_sec(alan_secenekleri).format(area_kategori_belirle(row['mean_area'])))
    cumleler.append(rastgele_cumle_sec(duzgunluk_secenekleri).format(smoothness_kategori_belirle(row['mean_smoothness'])))
    
    # Tanı için cümle ekleyin
    cumleler.append(rastgele_cumle_sec(tani_secenekleri[row['diagnosis']]))
    
    return cumleler

on_isleyici = OnIsle(dataset_path, tanimlayici_cumle_olustur)
on_isleyici.isle_kaydet()
