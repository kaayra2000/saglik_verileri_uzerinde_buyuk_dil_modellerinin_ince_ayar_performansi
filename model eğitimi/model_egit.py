import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.callbacks import EarlyStopping

from sabitler import *

# TensorFlow için uyumlu model ve tokenizer yükleme
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True, num_labels=2)

# Veri dosyasını oku
data_filepath = '../ana_veri_seti/islenmis_diabetes_prediction_dataset.csv'
df = pd.read_csv(data_filepath)

X = list(df["text"])
y = list(df["diabetes"])

# İlk olarak veriyi %80 eğitim, %20 doğrulama/test olarak ayır
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Daha sonra kalan %20'yi yarıya bölerek %10 doğrulama ve %10 test seti oluştur
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)

# Verilerin tokenleştirilmesi
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="tf")
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512, return_tensors="tf")
# Test verilerinin tokenleştirilmesi
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512, return_tensors="tf")

batch_size = 32
# TensorFlow Dataset oluşturma
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_test_tokenized),
    y_test
)).batch(batch_size)
# TensorFlow Dataset oluşturma
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_train_tokenized),
    y_train
)).shuffle(1000).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_val_tokenized),
    y_val
)).batch(batch_size)

# EarlyStopping geri arama işlevini tanımlayın
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Modeli eğitme
model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[early_stopping])
"""
# Modelin test verileri üzerinde değerlendirilmesi
# Modelin tahminlerini yapın
y_pred = model.predict(test_dataset)
# Tahminleri sınıf etiketlerine dönüştürün (örneğin, en yüksek olasılığa sahip sınıfı seçin)
y_pred_labels = np.argmax(y_pred)
# Gerçek sınıf etiketlerini alın
y_true_labels = y_test
# Precision (kesinlik) hesaplayın
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')

# Recall (duyarlılık) hesaplayın
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')

# F1 skoru hesaplayın
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

# Sonuçları yazdırın
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
"""
model.save_pretrained(model_adi)
tokenizer.save_pretrained(model_adi)