from collections import Counter
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, SORU_TOKEN, CEVAP_TOKEN, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.decoded_texts = []
        # Tokenizer için bir padding token'ı atayın, eğer yoksa
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.questions = self.data['text'].apply(lambda  x: SORU_TOKEN + " " + normalize_answer(x.split("Cevap:")[0].strip().replace("Soru:","")) + " ")
        self.answers = self.data['text'].apply(lambda  x: CEVAP_TOKEN + " " + normalize_answer(x.split("Cevap:")[1].strip()) + " ")

        self.inputs = []
        self.attention_masks = []

        for question, answer in zip(self.questions, self.answers):
            # Soru ve cevabı tokenizer ile encode edin
            encoded_pair = self.tokenizer.encode_plus(question, answer,
                                                      add_special_tokens=True,
                                                      max_length=max_len,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors="pt")
            # encodeanan değer nasıl decodelanıyor desti için
            #decoded_text = self.tokenizer.decode(encoded_pair['input_ids'][0], skip_special_tokens=False)
            #self.decoded_texts.append(decoded_text)
            self.inputs.append(encoded_pair['input_ids'])
            self.attention_masks.append(encoded_pair['attention_mask'])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx].squeeze(),  # Batch boyutunu kaldır
            'attention_mask': self.attention_masks[idx].squeeze()  # Batch boyutunu kaldır
        }

def normalize_answer(text):
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)

    # Tüm metni küçük harfe çevir
    text = text.lower()

    # Fazladan boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def compute_exact_match(prediction, truth):
    """Tahmini ve gerçeği normalize edip, birebir örtüşen karakterleri say."""
    normalized_prediction = normalize_answer(prediction)
    normalized_truth = normalize_answer(truth)

    # Kesişim kümesini kullanarak birebir örtüşen karakterleri bul
    # Önce her iki string'deki karakterleri say
    pred_char_counts = Counter(normalized_prediction)
    truth_char_counts = Counter(normalized_truth)

    # Kesişim kümesini hesapla ve ortak karakter sayısını bul
    common_chars = pred_char_counts & truth_char_counts
    num_same = sum(common_chars.values())

    # Birebir örtüşen karakter sayısını, en uzun string uzunluğuna bölerek oranını döndür
    return num_same / max(len(normalized_prediction), len(normalized_truth))

def compute_f1(prediction, truth):
    """Tahmini ve gerçeği normalize edip karakter bazında F1 skorunu hesapla."""
    # Metinleri normalize et ve karakter listelerine dönüştür.
    pred_chars = list(normalize_answer(prediction))
    truth_chars = list(normalize_answer(truth))

    # Ortak karakter sayısını bul
    common_chars = Counter(pred_chars) & Counter(truth_chars)
    num_same = sum(common_chars.values())

    # Doğruluk ve hatırlama hesapla
    if len(pred_chars) == 0 or len(truth_chars) == 0:
        return int(pred_chars == truth_chars)  # Eğer bir dizi boşsa ve diğeri de boşsa 1, değilse 0 döndür.

    precision = num_same / len(pred_chars)
    recall = num_same / len(truth_chars)

    # F1 skorunu hesapla
    if precision + recall == 0:
        return 0.0  # Eğer hem doğruluk hem de hatırlama 0 ise F1 skoru da 0 olur.

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
import re

def clean_special_tokens(text):
    """Metinden '<|' ile başlayıp '|>' ile biten özel tokenları kaldır."""
    return re.sub(r'<\|.*?\|>', '', text)


def evaluate_model(model, dataloader, device, tokenizer):
    model.eval()
    total_f1 = 0.0
    total_exact_match = 0.0
    total_count = 0
    special_tokens = [tokenizer.additional_special_tokens[0], tokenizer.additional_special_tokens[1]]

    with torch.no_grad():
        for batch in dataloader:
            # Batch'ten gelen verileri cihaza yükle
            inputs = {k: v.to(device) for k, v in batch.items()}
            # Decode inputs to text to split based on special token
            decoded_inputs = [tokenizer.decode(inp, skip_special_tokens=False) for inp in inputs['input_ids']]
            # Yeni metin üretimi ve işleme
            predicted_texts = []
            answers = []
            for text in decoded_inputs:
                # Metni special_tokens[1] ile böl ve öncesini kullan
                if special_tokens[1] in text:
                    parts = text.split(special_tokens[1])
                else:
                    parts = [text, text]
                prompt_part = parts[0]
                full_prompt_text = prompt_part + special_tokens[1]
                prompt_part = clean_special_tokens(prompt_part)
                generated_text = generate_eval_text(full_prompt_text, prompt_part, tokenizer, model, device)
                predicted_texts.append(generated_text)
                answers.append(parts[1].strip())
            
            answers = [clean_special_tokens(answer) for answer in answers]
            
            # F1 skoru ve tam eşleşme hesaplaması
            for pred_text, true_answer in zip(predicted_texts, answers):
                total_f1 += compute_f1(pred_text, true_answer)
                total_exact_match += compute_exact_match(pred_text, true_answer)

            total_count += len(inputs['input_ids'])
        print(predicted_texts[0])
        print(answers[0])
    average_f1 = total_f1 / total_count
    average_exact_match = total_exact_match / total_count

    return average_f1, average_exact_match

def plot_metrics(val_f1_list, val_exact_match_list, lr, bs, epochs, gorsel_yolu, gorsel_adi = "Validation"):
    os.makedirs(gorsel_yolu, exist_ok=True)  # Klasör yoksa
    # Kayıp grafiğini kaydetme yolu
    loss_graph_path = os.path.join(gorsel_yolu, f"{gorsel_adi}_lr_{lr}_bs_{bs}_loss.png")
    # Perplexity grafiğini kaydetme yolu
    perplexity_graph_path = os.path.join(gorsel_yolu, f"{gorsel_adi}_lr_{lr}_bs_{bs}_perplexity.png")

    epochs_range = range(1, epochs + 1)

    # Kayıp grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, val_f1_list, label='F1')
    plt.title(f'{gorsel_adi} F1 with lr={lr}, bs={bs}')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_graph_path)
    plt.close()  # Aktif figürü kapat

    # Perplexity grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, val_exact_match_list, label='Exact Match')
    plt.title(f'{gorsel_adi} Exact Match with lr={lr}, bs={bs}')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match')
    plt.legend()
    plt.tight_layout()
    plt.savefig(perplexity_graph_path)
    plt.close()  # Aktif figürü kapat


def find_unique_words(texts):
    # Tüm metinlerdeki kelimeleri bulun ve sayın
    words = Counter(re.findall(r'\w+', ' '.join(texts)))
    return list(words.keys())


def save_checkpoint(model, optimizer, epoch, checkpoint_path, batch_index, val_exact_match_list, val_f1_list,
                    checkpoint_name, is_print = False):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    batch_index = batch_index + 1
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_index': batch_index,
        'val_exact_match_list': val_exact_match_list,
        'val_f1_list': val_f1_list
    }
    tmp_checkpoint_path = os.path.join(checkpoint_path, f"tmp_{checkpoint_name}")
    final_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name)
    torch.save(checkpoint, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, final_checkpoint_path)
    if is_print:
      print(f"Checkpoint şuraya kaydedildi -> {checkpoint_path} epoch: {epoch}, batch: {batch_index}")
      print("EM Scores:")
      for em_score in val_exact_match_list:
          print(em_score)
      print("F1 Scores:")
      for f1_score in val_f1_list:
          print(f1_score)
def load_checkpoint(model, checkpoint_path, checkpoint_name, device):
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_name)
    # Checkpoint dosyasının varlığını kontrol et
    if not os.path.exists(checkpoint_filepath):
        print("Checkpoint bulunamadı")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer başlatma
        return model, optimizer, 0, 0, [], []
    print(f"Checkpoint şu konumdan yükleniyor ->  {checkpoint_filepath}")
    checkpoint = torch.load(checkpoint_filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer başlatma
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_index = checkpoint.get('batch_index', 0)
    val_exact_match_list = checkpoint.get('val_exact_match_list', [])
    val_f1_list = checkpoint.get('val_f1_list', [])

    print(f"Checkpoint başarıyla yüklendi: {checkpoint_path} (Epoch: {epoch}, Batch: {batch_index})")

    return model, optimizer, epoch, batch_index, val_exact_match_list, val_f1_list


def train(model, optimizer, start_epoch, start_batch_index,
          device, train_dataloader, validation_dataloader, val_exact_match_list, val_f1_list,
          tokenizer, checkpoint_path, epoch_sayisi, checkpoint_name, lr, bs, gorsel_yolu):
    train_dataloader_len_5 = len(train_dataloader) / 5
    for epoch in range(start_epoch, epoch_sayisi):
        print(f"Epoch {epoch} başlıyor...")
        model.train()
        data_iter = iter(train_dataloader)

        # İlk epoch için başlangıç batch'ine kadar olan batch'leri atla
        if epoch == start_epoch:
            for _ in range(start_batch_index):
                next(data_iter)
            print(f"{start_batch_index} batch atlandı")
        for batch_index, batch in tqdm(enumerate(data_iter, start=start_batch_index),
                                       total=len(train_dataloader) - start_batch_index,
                                       desc=f"Processing {epoch}"):
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            save_checkpoint(model, optimizer, epoch, checkpoint_path, batch_index, val_exact_match_list, val_f1_list,
                            checkpoint_name, batch_index % train_dataloader_len_5 == train_dataloader_len_5 - 1)
        val_f1, val_exact_match = evaluate_model(model, validation_dataloader, device, tokenizer)
        print(f"Epoch {epoch} tamamlandı. F1: {val_f1}, Exact Match: {val_exact_match}")
        val_exact_match_list.append(val_exact_match)
        val_f1_list.append(val_f1)
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path, -1, val_exact_match_list, val_f1_list,
                    checkpoint_name, True)
        start_batch_index = 0
    plot_metrics(val_f1_list, val_exact_match_list, lr, bs, epoch_sayisi, gorsel_yolu)
    return model


def generate_text(full_prompt_text, prompt_text, tokenizer, model, device, max_length=100):
    # Girdi metnini tokenize et ve attention mask oluştur
    encoded_input = tokenizer.encode(full_prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

    # Metin üretimi
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length + 20,  # Cevap için ekstra uzunluk
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text.replace(prompt_text,"")

def generate_eval_text(full_prompt_text, prompt_text, tokenizer, model, device, max_length=100):
    # Girdi metnini tokenize et ve attention mask oluştur
    encoded_input = tokenizer.encode(full_prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

    # Metin üretimi - Rastgelelik olmadan ve her zaman en yüksek skorlu sonuçları döndürmek için ayarlar
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length + 20,  # Cevap için ekstra uzunluk
        do_sample=False,  # Rastgele örnekleme yapma, en yüksek olasılıklı tokenları seç
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text.replace(prompt_text, "")


def test_et(additional_special_tokens, tokenizer, model, device):
    while True:
        # Kullanıcıdan soru girdisi alma
        prompt_text = input("Soru: ")
        if prompt_text == "exit":
            break
        # Tam soru metnini oluştur
        full_prompt_text = f"{additional_special_tokens[0]} {prompt_text} {additional_special_tokens[1]}"

        # Metin üretimi ve çıktının gösterilmesi
        generated_text = generate_text(full_prompt_text, prompt_text, tokenizer, model, device)
        print("Modelin ürettiği cevap:", generated_text)