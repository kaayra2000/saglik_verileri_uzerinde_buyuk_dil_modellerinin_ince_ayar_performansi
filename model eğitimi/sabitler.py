import os
import datetime
num_labels = 2
batch_size = 32
epoch_sayisi = 2

# Dosya adları
model_adi = "CustomModel"
gorsel_klasor_adi = "gorseller"
check_point_path = "checkpoints"
anlik_saat = datetime.datetime.now()
# Assuming 'anlik_saat' is a datetime object
anlik_saat_str = anlik_saat.strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
sonuclar_dosyasi = os.path.join("sonuclar",anlik_saat_str)
gorsel_yolu = os.path.join(sonuclar_dosyasi,gorsel_klasor_adi)
history_file_path = os.path.join(sonuclar_dosyasi,'model_degerlendirme_sonuclari.txt')