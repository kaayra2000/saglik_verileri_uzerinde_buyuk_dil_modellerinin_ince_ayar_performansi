import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from sabitler import *

model = BertForSequenceClassification.from_pretrained(model_adi)
tokenizer = BertTokenizer.from_pretrained(model_adi)
# text = "That was good point"
text = "The patient is a 50.0 year old Male. The patient has hypertension, does not have heart disease. Smoking history: current. BMI: 27.32. HbA1c level: 5.7. Blood glucose level: 260."
inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()
print(predictions)