import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sabitler import *

os.chdir(model_path_with_label)
model = TFBertForSequenceClassification.from_pretrained(model_adi, from_pt=False, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(model_adi, from_pt=False)

text = "The patient is a 50.0 year old Male. The patient has hypertension, does not have heart disease. Smoking history: current. BMI: 27.32. HbA1c level: 5.7. Blood glucose level: 260."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors='tf')  # 'tf' tensörlerini kullan
outputs = model(**inputs)
predictions = tf.nn.softmax(outputs.logits, axis=-1)
predictions = predictions.numpy()
print(predictions)