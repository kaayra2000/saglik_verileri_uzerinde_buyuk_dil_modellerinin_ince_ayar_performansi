import tensorflow as tf
from transformers import TFBertForMaskedLM, BertTokenizer
from sabitler import *

os.chdir(model_path_without_label)
model = TFBertForMaskedLM.from_pretrained(model_adi, from_pt=False)
tokenizer = BertTokenizer.from_pretrained(model_adi, from_pt=False)

text = "The patient is a 50.0 year old Male. The patient has hypertension, does not have heart disease. Smoking history: current. BMI: 27.32. HbA1c level: 5.7. Blood glucose level: 260."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf')  # 'tf' tensörlerini kullan
outputs = model(**inputs)
# Modelin logit çıktılarını al
logits = outputs.logits

# Softmax fonksiyonu uygula
softmax_logits = tf.nn.softmax(logits, axis=-1)

# Her token pozisyonu için en yüksek olasılığa sahip token'in ID'sini bul
predicted_token_ids = tf.argmax(softmax_logits, axis=-1)

# Token ID'lerini gerçek kelimelere çevir
predicted_tokens = [tokenizer.convert_ids_to_tokens(id) for id in predicted_token_ids.numpy()]

print(predicted_tokens)