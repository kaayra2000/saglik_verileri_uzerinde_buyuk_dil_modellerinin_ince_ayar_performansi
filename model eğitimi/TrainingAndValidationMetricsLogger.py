from keras.callbacks import Callback
import tensorflow as tf
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
class DummyCallback(Callback):
    def __init__(self, input_shape):
        super(DummyCallback, self).__init__()
        self.input_shape = input_shape
    def on_epoch_begin(self, epoch, logs=None):
        dummy_input = tf.random.uniform([1, *self.input_shape])  # input_shape, modelinizin girdi boyutlarına göre ayarlanmalıdır
        self.model(dummy_input)  # Modeli örnek girdi ile çağırma
