import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


model_name = "CustomModel"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
# text = "That was good point"
text = "The patient is a 54.0 year old Female. The patient does not have hypertension, does not have heart disease. Smoking history: No Info. BMI: 27.32. HbA1c level: 6.6. Blood glucose level: 80."
inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()
print(predictions)