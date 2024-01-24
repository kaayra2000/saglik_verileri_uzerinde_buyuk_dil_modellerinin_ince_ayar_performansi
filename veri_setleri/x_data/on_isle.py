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