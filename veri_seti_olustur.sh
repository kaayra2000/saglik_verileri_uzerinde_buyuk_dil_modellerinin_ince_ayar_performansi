#!/bin/bash

cd veri_setleri
python3 on_isle.py
echo "Veri seti oluşturuldu."
cd ..
python3 icerik_hesaplayici.py
echo "İçerik hesaplandı."
python3 train_test_bolustur.py
echo "Train ve test veri setleri parçalandı."
head -n 1001 shuffled_train.csv > ana_veri_seti_deneme.csv
echo "Deneme veri seti oluşturuldu."