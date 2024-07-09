import os
import sys
from datetime import datetime
from openai import OpenAI

# Dosya yolunu almak için gerekli fonksiyonu içe aktar
sys.path.append("..")
from fonksiyonlar import veri_yolu_al

# API anahtarını oku
with open("../../../api_key.txt", "r") as file:
    api_key = file.read().strip()

# OpenAI istemcisini başlat
client = OpenAI(api_key=api_key)

# Veri yolunu al
veri_yolu = veri_yolu_al()

# Klasördeki tüm dosyaları oluşturma tarihine göre al ve sırala
file_list = []
for root, dirs, files in os.walk(veri_yolu):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        file_stat = os.stat(file_path)
        creation_time = datetime.fromtimestamp(file_stat.st_ctime)
        file_list.append((file_path, creation_time))

# Dosyaları oluşturma tarihine göre sırala
file_list.sort(key=lambda x: x[1])

# Sıralanan dosyaları yükle
uploaded_files = []
for file_path, creation_time in file_list:
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
        uploaded_files.append(batch_input_file)

# Yüklenen dosya bilgilerini yaz
with open("batch_file_info.txt", "w") as file:
    for file_info in uploaded_files:
        file.write(str(file_info) + "\n")
