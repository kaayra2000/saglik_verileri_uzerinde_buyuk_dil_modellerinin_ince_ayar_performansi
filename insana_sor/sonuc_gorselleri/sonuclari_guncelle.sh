#!/bin/bash

# Proje kök dizinini bul
ROOT_DIR=$(pwd)
while [[ ! -d "$ROOT_DIR/sonuc_gorselleri" && "$ROOT_DIR" != "/" ]]; do
  ROOT_DIR=$(dirname "$ROOT_DIR")
done

if [[ "$ROOT_DIR" == "/" ]]; then
  echo "Proje dizini bulunamadı!"
  exit 1
fi

# sonuc_gorselleri dizinine git
cd "$ROOT_DIR/sonuc_gorselleri" || exit 1

# Google Form sonuçlarını çek
cd google_form || exit 1
python3 form_sonuclarini_cek.py

# Ana dizine geri dön
cd ..

# Sonuçları birleştir
python3 sonuclari_birlestir.py .

python3 sonuclari_resultsa_yaz.py
