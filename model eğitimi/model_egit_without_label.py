from transformers import TFBertForMaskedLM, BertTokenizer
from sabitler import *
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from plot_and_save import plot_and_save_metric
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
# Burada 'BUFFER_SIZE' veri setinizin boyutuna uygun bir değer olmalıdır.
BUFFER_SIZE = len(encodings.input_ids)

dataset = dataset.shuffle(BUFFER_SIZE)

# Veri setini bölme oranını belirle
# Örneğin, %80 eğitim ve %20 doğrulama seti için 0.8 kullanılır.
TRAIN_SPLIT = 0.8
train_size = int(TRAIN_SPLIT * BUFFER_SIZE)

# Eğitim ve doğrulama setlerini ayır
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# EarlyStopping geri arama işlevini tanımlayın
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

best_accuracy = 0
best_params = {}
best_model = None
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
    gorsel_yolu = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,gorsel_klasor_adi)
    os.makedirs(gorsel_yolu, exist_ok=True)
    # Modeli yeniden yükleme
    model = TFBertForMaskedLM.from_pretrained(model_name)

    # Eğitim veri setini batch boyutuna göre oluştur
    train_dataset_batched = train_dataset.shuffle(len(texts)).batch(batch_size)
    val_dataset_batched = val_dataset.batch(batch_size)

    # Modeli derle
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=model.loss)

    # Modeli eğit
    history = model.fit(train_dataset_batched, epochs=epoch_sayisi, validation_data=val_dataset_batched, callbacks = [model_checkpoint_callback, early_stopping])
    plot_and_save_metric(history, 'loss', 'loss_plot.png', gorsel_yolu)
    val_loss = history.history['val_loss'][-1]
    # En iyi modeli ve parametreleri kaydet
    if val_loss < best_loss:  # Düşük kayıp daha iyidir
        best_loss = val_loss
        best_params = params
        best_model = model
os.makedirs(sonuclar_dosyasi, exist_ok=True)
best_model.save_pretrained(model_adi)
tokenizer.save_pretrained(model_adi)