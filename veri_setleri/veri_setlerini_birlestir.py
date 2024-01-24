import os
import pandas as pd

# Klasör yolu
klasor_yolu = '.'

# Klasördeki tüm CSV dosyalarını bul
csv_dosyalari = [dosya for dosya in os.listdir(klasor_yolu) if dosya.endswith('.csv')]

# Tüm CSV dosyalarını bir DataFrame'e yükle ve birleştir
birlesik_df = pd.concat([pd.read_csv(os.path.join(klasor_yolu, dosya)) for dosya in csv_dosyalari])

# Birleştirilmiş veriyi yeni bir CSV dosyasına kaydet
birlesik_df.to_csv(os.path.join("..",klasor_yolu, 'ana_veri_seti.csv'), index=False)
