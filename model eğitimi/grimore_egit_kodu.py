import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from sabitler import *
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Veri dosyasını oku
df = pd.read_csv(data_filepath)
os.makedirs(model_path_without_label, exist_ok=True)
os.chdir(model_path_without_label)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        # Tokenizer için bir padding token'ı atayın, eğer yoksa
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inputs = [self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) for text in texts]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def calculate_perplexity(loss):
    return torch.exp(loss).item()
# Özel collate_fn fonksiyonu
def collate_fn(batch):
    # Tokenizer'ın pad_token_id'sini kullanarak padding yapın
    input_ids = [item["input_ids"].squeeze(0) for item in batch]  # squeeze(0) boyutunu kaldırır
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # Dikkat maskesi oluşturun (padding olan yerlerde 0, diğer yerlerde 1)
    attention_mask = torch.zeros_like(input_ids_padded)
    attention_mask[input_ids_padded != tokenizer.pad_token_id] = 1
    
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask}
def plot_metrics(losses, perplexities, lr, bs, epochs, gorsel_yolu):
    # Kayıp grafiğini kaydetme yolu
    loss_graph_path = os.path.join(gorsel_yolu, f"lr_{lr}_bs_{bs}_loss.png")
    # Perplexity grafiğini kaydetme yolu
    perplexity_graph_path = os.path.join(gorsel_yolu, f"lr_{lr}_bs_{bs}_perplexity.png")
    
    epochs_range = range(1, epochs + 1)

    # Kayıp grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, losses, label='Loss')
    plt.title(f'Loss with lr={lr}, bs={bs}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_graph_path)
    plt.close()  # Aktif figürü kapat

    # Perplexity grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, perplexities, label='Perplexity')
    plt.title(f'Perplexity with lr={lr}, bs={bs}')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(perplexity_graph_path)
    plt.close()  # Aktif figürü kapat

def train(dataset, device, new_tokens):
    best_loss = np.inf
    best_model_params = None
    best_model = None
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            print(f"Training with learning rate: {lr}, batch size: {bs}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            # Yeni token'ları tokenizer'a ekle
            if new_tokens:
                tokenizer.add_tokens(new_tokens)
                # Modeli yeni token'larla uyumlu hale getir
                model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model.train()
            optimizer = AdamW(model.parameters(), lr=lr)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)

            epoch_losses = []
            epoch_perplexities = []

            for epoch in range(epoch_sayisi):
                total_loss = 0
                total_perplexity = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_perplexity += calculate_perplexity(loss.item())

                epoch_loss = total_loss / len(dataloader)
                epoch_perplexity = total_perplexity / len(dataloader)
                epoch_losses.append(epoch_loss)
                epoch_perplexities.append(epoch_perplexity)

                print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Perplexity: {epoch_perplexity:.4f}")
            batch_dosya_adi = f"batch_size={bs},learning_rate={lr}" 
            gorsel_yolu = os.path.join(sonuclar_dosyasi,batch_dosya_adi ,gorsel_klasor_adi)
            os.makedirs(gorsel_yolu, exist_ok=True)
            # Plotting
            plot_metrics(epoch_losses, epoch_perplexities, lr, bs, epoch_sayisi, gorsel_yolu)

            # Check for best model
            if epoch_losses[-1] < best_loss:
                best_loss = epoch_losses[-1]
                best_model_params = {'lr': lr, 'bs': bs}
                best_model = model
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved with loss {best_loss}")

    return best_model_params, best_model

def find_unique_words(texts):
    # Tüm metinlerdeki kelimeleri bulun ve sayın
    words = Counter(re.findall(r'\w+', ' '.join(texts)))
    return list(words.keys())



texts = df[metin].tolist() 
tokenizer = AutoTokenizer.from_pretrained(model_name)


unique_words = find_unique_words(texts)

# Tokenizer'da olmayan kelimeleri tespit et
new_tokens = [word for word in unique_words if tokenizer.convert_tokens_to_ids(word) == tokenizer.unk_token_id]

dataset = TextDataset(texts, tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_params, model = train(dataset, device, new_tokens)

# Model ve tokenizer kaydetme
model.save_pretrained(model_adi)
dataset.tokenizer.save_pretrained(model_adi)
