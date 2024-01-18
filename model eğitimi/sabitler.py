import os
import datetime
num_labels = 2
param_grid = {
    'learning_rate': [0.01, 0.001, 0.1, 0.0001],  # 4 farklı öğrenme oranı
    'batch_size': [16, 32, 64, 128],  # 4 farklı batch boyutu
}

epoch_sayisi = 30

# anahatlar
metin = "text"
label = "diabetes"
model_name = "malhajar/Mistral-7B-v0.2-meditron-turkish"
veri_seti_adi = "islenmis_diabetes_prediction_dataset.csv"
label_yolu = "with_label"
labelsiz_yolu = "without_label"
# Veri seti dosya yolu
data_filepath = os.path.join("../ana_veri_seti", veri_seti_adi)
# Dosya adları
model_adi = "BestModelParameters"
model_path = model_name.replace("/", "_").replace("-", "_")
model_path_with_label = os.path.join(label_yolu, model_path)
model_path_without_label = os.path.join(labelsiz_yolu, model_path)
gorsel_klasor_adi = "gorseller"
check_point_path = "checkpoints"
anlik_saat = datetime.datetime.now()
# Assuming 'anlik_saat' is a datetime object
anlik_saat_str = anlik_saat.strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
sonuclar_dosyasi = os.path.join("sonuclar",anlik_saat_str)
model_degerlendirme_sonuclari = 'model_degerlendirme_sonuclari.txt'