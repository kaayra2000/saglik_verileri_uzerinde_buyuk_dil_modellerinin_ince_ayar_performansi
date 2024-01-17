import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from keras import metrics
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizer
from keras.callbacks import EarlyStopping
import os
from sabitler import *
from keras.callbacks import Callback
from sklearn.model_selection import ParameterGrid
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from plot_and_save import plot_and_save_metric

# TensorFlow için uyumlu tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained(model_name)
# Veri dosyasını oku
df = pd.read_csv(data_filepath)

os.makedirs(model_path_with_label, exist_ok=True)
os.chdir(model_path_with_label)

X = list(df[metin])
y = list(df[label])
y = to_categorical(y)
# İlk olarak veriyi %80 eğitim, %20 doğrulama/test olarak ayır
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Daha sonra kalan %20'yi yarıya bölerek %10 doğrulama ve %10 test seti oluştur
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)

# Verilerin tokenleştirilmesi
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="tf")
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512, return_tensors="tf")
# Test verilerinin tokenleştirilmesi
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512, return_tensors="tf")
# Precision ve Recall metriklerini tanımlama
precision_metric = metrics.Precision(name='precision')
recall_metric = metrics.Recall(name='recall')
f1_score_metric = metrics.F1Score(name='F1Score', average='weighted')
epsilon = tf.keras.backend.epsilon()

# EarlyStopping geri arama işlevini tanımlayın
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
class CustomMetricsCallback(Callback):
    def __init__(self, file_path):
        super(CustomMetricsCallback, self).__init__()
        self.file_path = file_path
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        self.history.append(logs)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'a') as file:
            file.write(f"Epoch {epoch + 1}\n")
            file.write(f"Train Loss: {logs.get('loss')}, Train Accuracy: {logs.get('accuracy')}\n")
            file.write(f"Train Precision: {logs.get('precision')}, Train Recall: {logs.get('recall')}\n")
            file.write(f"Train F1 Score: {logs.get('F1Score')}\n")
            file.write(f"Validation Loss: {logs.get('val_loss')}, Validation Accuracy: {logs.get('val_accuracy')}\n")
            file.write(f"Validation Precision: {logs.get('val_precision')}, Validation Recall: {logs.get('val_recall')}\n")
            file.write(f"Validation F1 Score: {logs.get('val_F1Score')}\n\n")      
best_accuracy = 0
best_params = {}
best_model = None
best_history_file_path = None
for params in ParameterGrid(param_grid):
    lr = params['learning_rate']
    batch_size = params['batch_size']
    batch_dosya_adi = f"batch_size={batch_size},learning_rate={lr}" 
    # Modeli yükleme
    model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True, num_labels=num_labels)
    # TensorFlow Dataset oluşturma
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_train_tokenized),
        y_train
    )).shuffle(1000).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_val_tokenized),
        y_val
    )).batch(batch_size)
    model_checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(check_point_path,batch_dosya_adi,'model_epoch_{epoch:02d}'),
    save_freq='epoch',
    save_weights_only=False,
    save_format='tf',  # SavedModel formatında kaydet
    verbose=1
    )
    # Modeli derleme
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', precision_metric, recall_metric, f1_score_metric]
    )
    # Özel callback oluşturun
    history_file_path = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,model_degerlendirme_sonuclari)
    gorsel_yolu = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,gorsel_klasor_adi)
    custom_metrics_callback = CustomMetricsCallback(history_file_path)
    # Modeli eğitme
    history = model.fit(train_dataset, epochs=epoch_sayisi, validation_data=val_dataset, callbacks=[early_stopping, custom_metrics_callback, model_checkpoint_callback])
    accuracy = history.history['val_accuracy'][-1]
    # Metrikleri Görselleştirme ve Kaydetme
    os.makedirs(gorsel_yolu, exist_ok=True)
    # Her bir metrik için görselleştirme ve kaydetme işlemini yapın
    plot_and_save_metric(custom_metrics_callback.history, 'accuracy', 'accuracy_plot.png', gorsel_yolu)
    plot_and_save_metric(custom_metrics_callback.history, 'loss', 'loss_plot.png', gorsel_yolu)
    plot_and_save_metric(custom_metrics_callback.history, 'precision', 'precision_plot.png',gorsel_yolu)
    plot_and_save_metric(custom_metrics_callback.history, 'recall', 'recall_plot.png', gorsel_yolu)
    plot_and_save_metric(custom_metrics_callback.history, 'F1Score', 'F1Score_plot.png', gorsel_yolu)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        best_model = model
        best_history_file_path = history_file_path

print("En iyi parametreler:", best_params)
with open(os.path.join(check_point_path, "best_params.txt"), 'w') as file:
    file.write(f"En iyi parametreler: {best_params}\n")

os.makedirs(sonuclar_dosyasi, exist_ok=True)
best_model.save_pretrained(model_adi)
tokenizer.save_pretrained(model_adi)

# TensorFlow Dataset oluşturma
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_test_tokenized),
    y_test
)).batch(best_params['batch_size'])

# Modelin test verileri üzerinde değerlendirilmesi
# Modelin tahminlerini yapın
y_pred = model.predict(test_dataset)
# Modelin tahminlerini alın ve logitleri kullanın
y_pred_logits = y_pred.logits
# Logitleri sınıf etiketlerine dönüştürün
y_pred_labels = np.argmax(y_pred_logits, axis=1)
# Gerçek sınıf etiketlerini alın
y_true_labels = np.argmax(y_test, axis=1)

# Precision, Recall ve F1 skor, accuracy ve loss değerlerini hesaplayın
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
accuracy = np.mean(y_pred_labels == y_true_labels)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_test, y_pred_logits).numpy()


# Sonuçları bir değişkene atayın
sonuclar = f"TEST\nPrecision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}, Loss: {loss}\n\n"
# Sonuçları bir dosyaya yazdırın
with open(best_history_file_path, 'a') as file:
    file.write(sonuclar)