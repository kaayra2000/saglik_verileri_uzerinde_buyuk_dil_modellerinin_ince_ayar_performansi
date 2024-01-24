import os
import sys
# Mevcut dosyanın bulunduğu dizinin tam yolunu al
mevcut_dizin = os.path.dirname(os.path.abspath(__file__))

# Bir üst dizinin yolunu bul
ust_dizin = os.path.abspath(os.path.join(mevcut_dizin, os.pardir))

# Bu yolu sys.path'e ekle
sys.path.append(ust_dizin)
from yuksek_dusuk import *
def radius_kategori_belirle(radius):
    # Bu değerler varsayımsaldır ve gerçek verilere göre ayarlanmalıdır.
    if radius < 14:
        return rastgele_cumle_sec(dusuk_alternatifler)
    elif radius < 20:
        return rastgele_cumle_sec(orta_alternatifler)
    else:
        return rastgele_cumle_sec(yuksek_alternatifler)

def texture_kategori_belirle(texture):
    # Bu değerler varsayımsaldır ve gerçek verilere göre ayarlanmalıdır.
    if texture < 15:
        return rastgele_cumle_sec(dusuk_alternatifler)
    elif texture < 20:
        return rastgele_cumle_sec(orta_alternatifler)
    else:
        return rastgele_cumle_sec(yuksek_alternatifler)

def perimeter_kategori_belirle(perimeter):
    # Bu değerler varsayımsaldır ve gerçek verilere göre ayarlanmalıdır.
    if perimeter < 90:
        return rastgele_cumle_sec(dusuk_alternatifler)
    elif perimeter < 130:
        return rastgele_cumle_sec(orta_alternatifler)
    else:
        return rastgele_cumle_sec(yuksek_alternatifler)
def area_kategori_belirle(area):
    # Bu değerler varsayımsaldır ve gerçek verilere göre ayarlanmalıdır.
    if area < 600:
        return rastgele_cumle_sec(dusuk_alternatifler)
    elif area < 1000:
        return rastgele_cumle_sec(orta_alternatifler)
    else:
        return rastgele_cumle_sec(yuksek_alternatifler)

def smoothness_kategori_belirle(smoothness):
    # Bu değerler varsayımsaldır ve gerçek verilere göre ayarlanmalıdır.
    if smoothness < 0.07:
        return rastgele_cumle_sec(dusuk_alternatifler)
    elif smoothness < 0.1:
        return rastgele_cumle_sec(orta_alternatifler)
    else:
        return rastgele_cumle_sec(yuksek_alternatifler)