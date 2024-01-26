from transformers import TFGPT2LMHeadModel, AutoTokenizer
from sabitler import *
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import os
from fonksiyonlar import *

# TensorFlow için uyumlu model ve tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Veri dosyasını oku
df = pd.read_csv(data_filepath)

os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)

texts = df[metin].tolist() 
# Verileri tokenleştirme
encodings = tokenizer(texts, padding='longest', truncation=True, return_tensors="tf", max_length=512)

train_dataset, val_dataset, test_dataset  = veri_seti_parcala(encodings)

callbacks = get_callbacks()
best_loss = float('inf')
best_params = {}
best_model = None
best_history_file_path = None

for params in ParameterGrid(param_grid):
    lr = params['learning_rate']
    batch_size = params['batch_size']
    batch_dosya_adi = f"batch_size={batch_size},learning_rate={lr}" 
    # ModelCheckpoint callback
    #callbacks[0].filepath = os.path.join(check_point_path,batch_dosya_adi,'model_epoch_{epoch:02d}')
    history_file_path = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,model_degerlendirme_sonuclari)
    callbacks[1].log_file = history_file_path
    gorsel_yolu = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,gorsel_klasor_adi)
    os.makedirs(gorsel_yolu, exist_ok=True)
    # Modeli yeniden yükleme
    model = TFGPT2LMHeadModel.from_pretrained(model_name)

    # Eğitim veri setini batch boyutuna göre oluştur
    train_dataset_batched = train_dataset.shuffle(len(texts)).batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)
    # Modeli eğit
    original_data = modeli_egit(model, epoch_sayisi, train_dataset_batched, val_dataset_batched, callbacks, lr)
    plot_kaydet(original_data, gorsel_yolu)
    val_loss = original_data['val_loss'][-1]
    # En iyi modeli ve parametreleri kaydet
    best_loss, best_params, best_model, best_history_file_path = parametre_guncelle(val_loss, best_loss, best_params, best_model, model, best_history_file_path, history_file_path, params)

os.makedirs(sonuclar_dosyasi, exist_ok=True)
best_model.save_pretrained(model_adi)
tokenizer.save_pretrained(model_adi)
test_dataset_batched = test_dataset.batch(best_params['batch_size'])
# Modeli değerlendir
result = best_model.evaluate(test_dataset_batched)
# Sonuçları dosyaya yazmak için metin oluştur
sonuclar = f"Test Loss: {result[0]}, Test Accuracy: {result[1]}\n"
# Sonuçları bir dosyaya yazdırın
with open(best_history_file_path, 'a') as file:
    file.write(sonuclar)
