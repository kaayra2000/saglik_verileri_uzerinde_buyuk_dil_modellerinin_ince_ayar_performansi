import os
import datetime
num_labels = 2
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64]
}
epoch_sayisi = 3

# Dosya adları
model_adi = "BestModelParameters"
gorsel_klasor_adi = "gorseller"
check_point_path = "checkpoints"
anlik_saat = datetime.datetime.now()
# Assuming 'anlik_saat' is a datetime object
anlik_saat_str = anlik_saat.strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
sonuclar_dosyasi = os.path.join("sonuclar",anlik_saat_str)
gorsel_yolu = os.path.join(sonuclar_dosyasi,gorsel_klasor_adi)
model_degerlendirme_sonuclari = 'model_degerlendirme_sonuclari.txt'