from transformers import TFBertForMaskedLM, BertTokenizer
from sabitler import *
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import os
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


best_accuracy = 0
best_params = {}
best_model = None
for params in ParameterGrid(param_grid):
    lr = params["learning_rate"]
    bs = params["batch_size"]
    # Modeli yükleme
    model = TFBertForMaskedLM.from_pretrained(model_name)
    # Eğitim veri setini batch boyutuna göre oluştur
    train_dataset_batched = train_dataset.shuffle(len(texts)).batch(bs)
    val_dataset_batched = val_dataset.batch(bs)

    # Modeli derle
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=model.loss, metrics=['accuracy'])

    # Modeli eğit
    history = model.fit(train_dataset_batched, epochs=epoch_sayisi, validation_data=val_dataset_batched)

    # Doğrulama seti üzerindeki en iyi accuracy değerini al
    val_accuracy = max(history.history['val_accuracy'])

    print(f"Validation Accuracy: {val_accuracy}")

    # En iyi modeli ve parametreleri kaydet
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params
        best_model = model