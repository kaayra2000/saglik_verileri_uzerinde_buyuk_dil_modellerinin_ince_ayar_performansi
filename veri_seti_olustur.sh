#!/bin/bash

# Veri setleri klasörüne geçiş yap
echo "Veri setleri klasörüne geçiliyor..."
cd veri_setleri
if [ $? -ne 0 ]; then
    echo "Veri setleri klasörüne geçiş başarısız oldu!"
    exit 1
fi

# Ön işleme script'ini çalıştır
echo "Ön işleme yapılıyor..."
python3 on_isle.py
if [ $? -ne 0 ]; then
    echo "Ön işleme sırasında bir hata oluştu!"
    exit 1
fi

# Ana klasöre geri dön
echo "Ana klasöre geri dönülüyor..."
cd ..
if [ $? -ne 0 ]; then
    echo "Ana klasöre dönüş başarısız oldu!"
    exit 1
fi

echo "İçerik hesaplanıyor..."
python3 icerik_hesaplayici.py
if [ $? -ne 0 ]; then
    echo "İçerik hesaplama sırasında bir hata oluştu!"
    exit 1
fi

echo "Eğitim ve test verileri bölüştürülüyor..."
python3 train_test_bolustur.py
if [ $? -ne 0 ]; then
    echo "Eğitim ve test verilerinin bölüştürülmesi sırasında bir hata oluştu!"
    exit 1
fi

echo "İşlemler başarıyla tamamlandı!"
