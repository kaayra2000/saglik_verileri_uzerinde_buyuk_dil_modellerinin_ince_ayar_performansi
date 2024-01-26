import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from TrainingAndValidationMetricsLogger import *
def plot_and_save_metric(history, metric_name, file_name, gorsel_yolu):
    epochs = range(1, len(history) + 1)
    train_values = [log.get(metric_name) for log in history]
    val_values = [log.get('val_' + metric_name) for log in history]

    plt.figure()
    plt.plot(epochs, train_values, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_values, 'r*-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.savefig(os.path.join(gorsel_yolu, file_name))
    plt.close()


def modeli_egit(model,epoch_sayisi, train_dataset_batched, val_dataset_batched, callbacks, lr):
    # Modeli derle
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # Modeli eğitme
    return model.fit(train_dataset_batched, validation_data=val_dataset_batched, epochs=epoch_sayisi, callbacks=callbacks).history

def plot_kaydet(original_data, gorsel_yolu):
    new_format_data = []
    for i in range(len(original_data['loss'])):
        new_entry = {'loss': original_data['loss'][i], 'val_loss': original_data['val_loss'][i]}
        new_format_data.append(new_entry)
    plot_and_save_metric(new_format_data, 'loss', 'loss_plot.png', gorsel_yolu)
def parametre_guncelle(val_loss, best_loss, best_params, best_model, model, best_history_file_path, history_file_path, params):
    # En iyi modeli ve parametreleri kaydet
    if val_loss < best_loss:  # Düşük kayıp daha iyidir
        return val_loss, params, model, history_file_path
    else:
        return best_loss, best_params, best_model, best_history_file_path
    
def get_callbacks():
    """
    cp = ModelCheckpoint(
        filepath='model_epoch_{epoch:02d}',
        save_freq='epoch',
        save_weights_only=False,
        save_format='tf',  # SavedModel formatında kaydet
        verbose=1
        )"""
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tvml = TrainingAndValidationMetricsLogger("")
    return [es, tvml]



def veri_seti_parcala(encodings):
    # Veri setini oluştur (token_type_ids özelliğini çıkararak)
    dataset = tf.data.Dataset.from_tensor_slices((encodings.input_ids, encodings.attention_mask))
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
    return train_dataset, val_dataset, test_dataset
