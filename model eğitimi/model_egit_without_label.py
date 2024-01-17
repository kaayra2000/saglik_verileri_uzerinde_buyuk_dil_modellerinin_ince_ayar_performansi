from transformers import TFBertForMaskedLM, BertTokenizer
from sabitler import *
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from plot_and_save import plot_and_save_metric
from keras.callbacks import Callback

class TrainingAndValidationMetricsLogger(Callback):
    def __init__(self, log_file):
        super(TrainingAndValidationMetricsLogger, self).__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        # Eğitim seti metriklerini al
        train_accuracy = logs.get('accuracy')
        train_loss = logs.get('loss')
        # Doğrulama seti metriklerini al
        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        # Metrikleri dosyaya yaz
        with open(self.log_file, 'a') as log_file:
            log_file.write(f"Epoch {epoch+1}: Training Accuracy = {train_accuracy}, Training Loss = {train_loss}\n")
            log_file.write(f"Epoch {epoch+1}: Validation Accuracy = {val_accuracy}, Validation Loss = {val_loss}\n")

# TensorFlow için uyumlu model ve tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained(model_name)
# Veri dosyasını oku
df = pd.read_csv(data_filepath)

os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)

texts = df[metin].tolist() 
# Verileri tokenleştirme
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="tf", max_length=512)

# Veri setini oluştur
dataset = tf.data.Dataset.from_tensor_slices((encodings.input_ids, encodings.attention_mask, encodings.token_type_ids))
# Toplam veri boyutunu hesapla
total_size = len(encodings.input_ids)

# Eğitim ve test/doğrulama seti için bölme oranını belirle
TRAIN_SPLIT = 0.8
# Eğitim seti boyutunu hesapla
train_size = int(TRAIN_SPLIT * total_size)

# Eğitim ve test/doğrulama setlerini ayır
train_dataset = dataset.take(train_size)
test_val_dataset = dataset.skip(train_size)

# Test/doğrulama setinin boyutunu hesapla
test_val_size = total_size - train_size

# Test ve doğrulama seti için bölme oranını belirle (bu durumda her ikisi için %50)
TEST_VAL_SPLIT = 0.5
# Test seti boyutunu hesapla
test_size = int(TEST_VAL_SPLIT * test_val_size)

# Test ve doğrulama setlerini ayır
test_dataset = test_val_dataset.take(test_size)
val_dataset = test_val_dataset.skip(test_size)

# EarlyStopping geri arama işlevini tanımlayın
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

best_loss = float('inf')
best_params = {}
best_model = None
best_history_file_path = None
for params in ParameterGrid(param_grid):
    lr = params['learning_rate']
    batch_size = params['batch_size']
    batch_dosya_adi = f"batch_size={batch_size},learning_rate={lr}" 
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(check_point_path,batch_dosya_adi,'model_epoch_{epoch:02d}'),
        save_freq='epoch',
        save_weights_only=False,
        save_format='tf',  # SavedModel formatında kaydet
        verbose=1
        )
    history_file_path = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,model_degerlendirme_sonuclari)
    custom_metrics_callback = TrainingAndValidationMetricsLogger(history_file_path)
    gorsel_yolu = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,gorsel_klasor_adi)
    os.makedirs(gorsel_yolu, exist_ok=True)
    # Modeli yeniden yükleme
    model = TFBertForMaskedLM.from_pretrained(model_name, from_pt=True)

    # Eğitim veri setini batch boyutuna göre oluştur
    train_dataset_batched = train_dataset.shuffle(len(texts)).batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)

    # Modeli derle
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Modeli eğit
    history = model.fit(train_dataset_batched, epochs=epoch_sayisi, validation_data=val_dataset_batched, callbacks = [custom_metrics_callback,model_checkpoint_callback, early_stopping])
    new_format_data = []
    original_data = history.history
    for i in range(len(original_data['loss'])):
        new_entry = {'loss': original_data['loss'][i], 'val_loss': original_data['val_loss'][i]}
        new_format_data.append(new_entry)
    plot_and_save_metric(new_format_data, 'loss', 'loss_plot.png', gorsel_yolu)
    val_loss = original_data['val_loss'][-1]
    # En iyi modeli ve parametreleri kaydet
    if val_loss < best_loss:  # Düşük kayıp daha iyidir
        best_loss = val_loss
        best_params = params
        best_model = model
        best_history_file_path = history_file_path

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
