from collections import Counter
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
diyabet_var = [
    "Ben diyabet hastasıyım.",
    "Bende diyabet hastalığı mevcut."
]

diyabet_yok = [
    "Ben diyabet hastası değilim.",
    "Bende diyabet hastalığı mevcut değil."
]

VERI_BASLANGIC = "Soru:"
VERI_BITIS = "Cevap:"
TP = 0
FP = 1
FN = 2
TN = 3


class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, SORU_TOKEN, CEVAP_TOKEN, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.decoded_texts = []
        # Tokenizer için bir padding token'ı atayın, eğer yoksa
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.questions = self.data['text'].apply(lambda  x: SORU_TOKEN + " " + normalize_answer(x.split(VERI_BITIS)[0].strip().replace(VERI_BASLANGIC,"")))
        self.answers = self.data['text'].apply(lambda  x: CEVAP_TOKEN + " " + normalize_answer(x.split(VERI_BITIS)[1].strip()))

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
def longest_common_subsequence(s1:str, s2:str):
    """
    İki dize arasındaki en uzun ortak alt diziyi (LCS) hesaplar ve döndürür.

    Bu fonksiyon, iki dize (s1 ve s2) arasındaki en uzun ortak alt diziyi bulmak için
    dinamik programlama kullanır. LCS problemi, iki dizi arasındaki en uzun ortak eleman
    dizisini bulmak için kullanılan klasik bir algoritmadır ve bu algoritma, iki dizi
    arasındaki benzerlik derecesini ölçmek için kullanılabilir.

    Args:
        s1 (str): İlk dize.
        s2 (str): İkinci dize.

    Returns:
        str: İki dize arasındaki en uzun ortak alt dizi.

    Method:
        - Dinamik programlama tablosu (dp) kullanılarak her iki dizinin her bir elemanı
          karşılaştırılır ve LCS uzunluğu için bir tablo oluşturulur.
        - Tablo, LCS'in uzunluğunu depolar ve sonra bu tabloyu kullanarak LCS'in kendisi
          geriye dönük olarak çıkarılır.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # LCS'i geriye dönük olarak çıkarma
    lcs = []
    while m > 0 and n > 0:
        if s1[m - 1] == s2[n - 1]:
            lcs.append(s1[m - 1])
            m -= 1
            n -= 1
        elif dp[m - 1][n] >= dp[m][n - 1]:
            m -= 1
        else:
            n -= 1

    return ''.join(reversed(lcs))




def compute_exact_match_words_truth_array(prediction:str, truth_array:list):
    """
    Verilen bir tahmin metni ile birden fazla gerçek metin arasında kelime tabanlı Exact Match (EM) skorunu hesaplar
    ve en yüksek skoru sağlayan gerçek metni ve bu skoru döndürür.

    Bu fonksiyon, tahmin edilen metni, verilen gerçek metinler dizisiyle karşılaştırarak her bir eşleşme için
    kelime tabanlı Exact Match skorunu hesaplar. En yüksek skoru veren metin ve bu skor döndürülür, böylece
    tahminin hangi gerçek metinle en iyi performansı gösterdiğini belirlemek mümkün olur. Bu, özellikle
    birden fazla doğru cevabın mümkün olduğu durumlarda faydalı bir işlemdir.

    Args:
        prediction (str): Tahmin edilen metin.
        truth_array (list of str): Gerçek metinlerin listesi.

    Returns:
        tuple: En yüksek Exact Match skoru ve bu skoru sağlayan gerçek metin.
    """
    max_exact_match = 0
    max_truth = ""
    for truth in truth_array:
        exact_match = compute_exact_match_words(prediction, truth)
        if exact_match > max_exact_match:
            max_exact_match = exact_match
            max_truth = truth
    return max_exact_match, max_truth

def compute_exact_match_words(prediction:str, truth:str):
    """
    Verilen iki metin arasında kelime tabanlı Exact Match (EM) skoru hesaplar.
    Bu işlem, metinlerin normalleştirilmesi, kelimelere ayrılması ve kelime tekrarlarının sayılması ile başlar.
    Ardından, her iki metinde de bulunan kelimelerin minimum tekrar sayısı toplanır ve bu sayı,
    metinlerdeki toplam kelime sayısının minimumuna bölünerek bir oran hesaplanır.
    
    Kelimelerin sırası bu skorlamada dikkate alınmaz, sadece ortak kelimelerin sayısı önemlidir.
    Bu yöntem, metinler arasındaki kelime bazında benzerliği değerlendirmek için kullanışlıdır,
    özellikle metinler arası anlam benzerliğinin genel bir ölçümünü sağlamak istendiğinde tercih edilir.

    Args:
        prediction (str): Tahmin edilen metin.
        truth (str): Gerçek metin.

    Returns:
        float: Hesaplanan kelime tabanlı EM skoru. Eğer metinlerden biri boş ise veya hiçbir kelime eşleşmezse 0 olarak döner.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_truth = normalize_answer(truth)

    # Her iki metni kelimelere ayır ve say
    pred_words = Counter(normalized_prediction.split())
    truth_words = Counter(normalized_truth.split())

    # Ortak kelimelerin minimum tekrar sayısını hesapla
    common_words = pred_words & truth_words  # Kesişim kümesini bul
    common_min_counts = sum(min(pred_words[word], truth_words[word]) for word in common_words)

    # Prediction ve truth metinlerindeki toplam kelime sayısını hesapla
    pred_total_count = sum(pred_words.values())
    truth_total_count = sum(truth_words.values())

    # Minimum toplam kelime sayısı
    min_total_count = min(pred_total_count, truth_total_count)

    # Ortak kelimelerin minimum tekrar sayısının, en az kelime sayısına oranını hesapla
    if min_total_count > 0:
        score = common_min_counts / min_total_count
    else:
        score = 0  # Eğer metinlerden biri boş ise skor 0 olur

    return score

def compute_lcs_letter_truth_array(prediction:str, truth_array:list):
    """
    Bir tahmin metni ile birden çok gerçek metin arasındaki en yüksek F1 skorunu hesaplar.
    
    Bu fonksiyon, verilen bir tahmin metnini bir gerçek metinler dizisiyle karşılaştırarak,
    her bir eşleşme için harf tabanlı LCS yöntemi kullanarak F1 skorunu hesaplar. 
    Hesaplanan tüm F1 skorları arasından en yüksek olanı seçilir ve döndürülür.
    
    Fonksiyon, birden fazla doğru metin arasında tahmin metnin hangisiyle en iyi performansı
    gösterdiğini belirlemek için kullanışlıdır. Bu, özellikle birden fazla doğru cevabın mümkün
    olduğu durumlarda kullanılabilir.

    Args:
        prediction (str): Tahmin edilen metin.
        truth_array (list of str): Gerçek metinlerin listesi.

    Returns:
        float: En yüksek hesaplanan F1 skoru. Eğer truth_array boşsa veya hiçbir metin
        eşleşme göstermezse 0 olarak döner.
    """
    max_f1 = 0
    for truth in truth_array:
        f1 = compute_lcs_letter(prediction, truth)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1

def compute_lcs_letter(prediction:str, truth:str):
    """
    Verilen iki metin arasında, harflerin sırasını dikkate alarak en uzun ortak alt dizi (LCS) skorunu hesaplar.
    Bu işlem, metinlerin benzerlik derecesini değerlendirmek için kullanılır ve karakter bazında bir analiz sağlar.

    Normalizasyon işlemi metinlerin küçük harfe çevrilmesi, noktalama işaretlerinin kaldırılması ve 
    fazladan boşlukların temizlenmesi gibi adımları içerir. Ardından, dinamik programlama kullanılarak 
    her iki metin arasındaki en uzun ortak ardışık karakter dizisi bulunur.

    Hesaplanan LCS uzunluğu, her iki metnin uzunluğuna bölünerek doğruluk (precision) ve hatırlama (recall) oranları elde edilir.
    Bu oranlar, metnin ne kadarının diğer metinle eşleştiğini gösterir. Son olarak, bu iki oranın harmonik ortalaması olan 
    F1 skoru hesaplanır.

    Args:
        prediction (str): Tahmin edilen metin.
        truth (str): Gerçek metin.

    Returns:
        float: Hesaplanan F1 skoru. Eğer precision ve recall'ın toplamı 0 ise, F1 skoru da 0 olarak döner.
    """
    # Metinleri normalize et
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(truth)

    # En uzun ortak alt diziyi bul
    lcs = longest_common_subsequence(pred_norm, truth_norm)
    num_same = len(lcs)

    # Doğruluk ve hatırlama hesapla
    if len(pred_norm) == 0 or len(truth_norm) == 0:
        return int(pred_norm == truth_norm)

    precision = num_same / len(pred_norm)
    recall = num_same / len(truth_norm)

    # F1 skorunu hesapla
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_disease(prediction:str, truth:str):
    """
    Verilen tahmin ve gerçek metinlerini değerlendirerek, bu metinlerin 'diyabet var' veya 'diyabet yok'
    durumlarına ne kadar uyduğunu analiz eder ve sonuçları döndürür.

    Bu fonksiyon, tahmin ve gerçek metinleri belirlenmiş 'diyabet var' ve 'diyabet yok' kelimeleri listelerine göre 
    değerlendirir ve her bir metin için bu kategorilere ait skorları ve eşleşen metinleri hesaplar. 
    Daha sonra, bu bilgileri kullanarak tahmin edilen ve gerçek durumları karşılaştırır. 
    Eğer tahmin edilen durum gerçek durumla uyuşuyorsa (hem tahmin hem de gerçek için 'diyabet var' veya 'diyabet yok'),
    ilgili durumu (True Positive veya True Negative) döndürür. Aksi takdirde, yanlış pozitif veya yanlış negatif
    olarak sınıflandırır.

    Args:
        prediction (str): Tahmin edilen metin.
        truth (str): Gerçek metin.

    Returns:
        tuple: (durum, prediction_metin, truth_metin)
            - durum (int): Tahminin durumunu belirten sabit (TP, TN, FP, FN).
            - prediction_metin (str): Tahmine dayalı eşleşen metin parçası.
            - truth_metin (str): Gerçeğe dayalı eşleşen metin parçası.
    """
    prediction_metin = ""
    truth_metin = ""
    durum = 0
    prediction = normalize_answer(prediction)
    truth = normalize_answer(truth)
    prediction_diyabet_var_skor, prediction_diyabet_var_metin = compute_exact_match_words_truth_array(prediction, diyabet_var)
    prediction_diyabet_yok_skor, prediction_diyabet_yok_metin = compute_exact_match_words_truth_array(prediction, diyabet_yok)
    truth_diyabet_var_skor, truth_diyabet_var_metin = compute_exact_match_words_truth_array(truth, diyabet_var)
    truth_diyabet_yok_skor, truth_diyabet_yok_metin = compute_exact_match_words_truth_array(truth, diyabet_yok)
    prediction_diyabet = True if prediction_diyabet_var_skor > prediction_diyabet_yok_skor else False
    truth_diyabet = True if truth_diyabet_var_skor > truth_diyabet_yok_skor else False
    if prediction_diyabet == truth_diyabet:
        if prediction_diyabet:
            prediction_metin = prediction_diyabet_var_metin
            truth_metin = truth_diyabet_var_metin
            durum = TP
        else:
            prediction_metin = prediction_diyabet_yok_metin
            truth_metin = truth_diyabet_yok_metin
            durum = TN
    elif prediction_diyabet:
        prediction_metin = prediction_diyabet_var_metin
        truth_metin = truth_diyabet_yok_metin
        durum = FP
    else:
        prediction_metin = prediction_diyabet_yok_metin
        truth_metin = truth_diyabet_var_metin
        durum = FN
    return durum, prediction_metin, truth_metin
def f1_metrikleri_guncelle(disease_durumu:int, total_tp:int, total_fp:int, total_fn:int, total_tn:int):
    """
    Hastalık durumuna göre F1 metriklerini (TP, FP, FN, TN) günceller.
    
    Bu fonksiyon, bir hastalık durumu sonucuna (TP, FP, FN, TN) göre 
    ilgili metrikleri günceller. Sonuç, modelin tahminlerinin doğruluğunu
    değerlendirmede kullanılan dört ana kategoriden biridir:
    
    - TP (True Positive): Modelin doğru olarak pozitif tahminde bulunduğu durum sayısı.
    - FP (False Positive): Modelin yanlış olarak pozitif tahminde bulunduğu durum sayısı.
    - FN (False Negative): Modelin yanlış olarak negatif tahminde bulunduğu durum sayısı.
    - TN (True Negative): Modelin doğru olarak negatif tahminde bulunduğu durum sayısı.
    
    Args:
        disease_durumu (str): Hastalığın durumu (TP, FP, FN, TN olarak belirtilir).
        total_tp (int): Mevcut True Positive sayısı.
        total_fp (int): Mevcut False Positive sayısı.
        total_fn (int): Mevcut False Negative sayısı.
        total_tn (int): Mevcut True Negative sayısı.

    Returns:
        tuple: Güncellenmiş (total_tp, total_fp, total_fn, total_tn) değerlerini içeren bir dörtli (tuple).
    """
    if disease_durumu == TP:
        total_tp += 1
    elif disease_durumu == FP:
        total_fp += 1
    elif disease_durumu == FN:
        total_fn += 1
    else:
        total_tn += 1
    return total_tp, total_fp, total_fn, total_tn

def clean_special_tokens(text):
    """Metinden '<|' ile başlayıp '|>' ile biten özel tokenları kaldır."""
    return re.sub(r'<\|.*?\|>', '', text)
def compute_f1_score(tp:int, fp:int, fn:int):
    """
    TP, TN, FP, ve FN değerlerini alarak F1 skorunu hesaplar.

    Args:
        tp (int): True Positive sayısı.
        tn (int): True Negative sayısı.
        fp (int): False Positive sayısı.
        fn (int): False Negative sayısı.

    Returns:
        float: Hesaplanan F1 skoru. Eğer precision ve recall'ın toplamı 0 ise F1 skoru 0 döner.
    """
    # Precision ve recall hesapla
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    # F1 skorunu hesapla
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score



def evaluate_model(model, dataloader, device, tokenizer):
    model.eval()
    total_lcs = 0.0
    total_exact_match = 0.0
    total_count = 0
    special_tokens = [tokenizer.additional_special_tokens[0], tokenizer.additional_special_tokens[1]]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
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
                answers.append(clean_special_tokens(parts[1].strip()))
            
            
            # F1 skoru ve tam eşleşme hesaplaması
            for pred_text, true_answer in zip(predicted_texts, answers):
                disease_durumu, _, en_yakin_dogru_cevap = compute_disease(pred_text, true_answer)
                total_tp, total_fp, total_fn, total_tn = f1_metrikleri_guncelle(disease_durumu, total_tp, total_fp, total_fn, total_tn)
                total_lcs += compute_lcs_letter(pred_text, en_yakin_dogru_cevap)
                total_exact_match += compute_exact_match_words(pred_text, en_yakin_dogru_cevap)
               

            total_count += len(inputs['input_ids'])
        print(predicted_texts[0])
        print(answers[0])
    average_lcs = total_lcs / total_count
    f1_score = compute_f1_score(total_tp, total_fp, total_fn)
    average_exact_match = total_exact_match / total_count
    accuracy = (total_tp + total_tn) / total_count 
    return accuracy, average_exact_match, f1_score, average_lcs 

def plot_metrics(plot_list, lr, bs, epochs:int, gorsel_yolu:str, y_label:str, gorsel_adi = "Validation"):
    os.makedirs(gorsel_yolu, exist_ok=True)  # Klasör yoksa
    # Kayıp grafiğini kaydetme yolu
    loss_graph_path = os.path.join(gorsel_yolu, f"{gorsel_adi}_lr_{lr}_bs_{bs}_{y_label}.png")

    epochs_range = range(1, epochs + 1)

    # Kayıp grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, plot_list, label='F1')
    plt.title(f'{gorsel_adi} F1 with lr={lr}, bs={bs}')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_graph_path)
    plt.close()  # Aktif figürü kapat


def find_unique_words(texts):
    # Tüm metinlerdeki kelimeleri bulun ve sayın
    words = Counter(re.findall(r'\w+', ' '.join(texts)))
    return list(words.keys())

def print_metric(metric_list:list, title:str):
    print(f"{title} Scores:")
    for score in metric_list:
        print(score)
def save_checkpoint(model, optimizer, epoch, checkpoint_path, batch_index,
                    checkpoint_name,val_accuracy_list:list, val_exact_match_list:list,
                    val_f1_score_list:list, var_lcs_list:list,
                    is_print = False):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    batch_index = batch_index + 1
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_index': batch_index,
        'val_accuracy_list':val_accuracy_list,
        'val_exact_match_list': val_exact_match_list,
        'val_f1_score_list': val_f1_score_list,
        'var_lcs_list':var_lcs_list
        
    }
    tmp_checkpoint_path = os.path.join(checkpoint_path, f"tmp_{checkpoint_name}")
    final_checkpoint_path = os.path.join(checkpoint_path, checkpoint_name)
    torch.save(checkpoint, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, final_checkpoint_path)
    if is_print:
      print(f"Checkpoint şuraya kaydedildi -> {checkpoint_path} epoch: {epoch}, batch: {batch_index}")
      print_metric(val_accuracy_list, "Accuracy")
      print_metric(val_exact_match_list, "EM")
      print_metric(val_f1_score_list, "F1")
      print_metric(var_lcs_list, "LCS")
      print("EM Scores:")

def load_checkpoint(model, checkpoint_path, checkpoint_name, device):
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_name)
    # Checkpoint dosyasının varlığını kontrol et
    if not os.path.exists(checkpoint_filepath):
        print("Checkpoint bulunamadı")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer başlatma
        return model, optimizer, 0, 0, [], [], [], []
    print(f"Checkpoint şu konumdan yükleniyor ->  {checkpoint_filepath}")
    checkpoint = torch.load(checkpoint_filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer başlatma
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_index = checkpoint.get('batch_index', 0)
    val_accuracy_list = checkpoint.get('val_accuracy_list', [])
    val_exact_match_list = checkpoint.get('val_exact_match_list', [])
    val_f1_score_list = checkpoint.get('val_f1_score_list', [])
    var_lcs_list = checkpoint.get('var_lcs_list', [])


    print(f"Checkpoint başarıyla yüklendi: {checkpoint_path} (Epoch: {epoch}, Batch: {batch_index})")

    return model, optimizer, epoch, batch_index, val_accuracy_list, val_exact_match_list, val_f1_score_list, var_lcs_list


def train(model, optimizer, start_epoch, start_batch_index,
          device, train_dataloader, validation_dataloader,
          tokenizer, checkpoint_path, epoch_sayisi, checkpoint_name, lr, bs, gorsel_yolu,
          val_accuracy_list:list, val_exact_match_list:list, val_f1_score_list:list, var_lcs_list:list):
    train_dataloader_len_5 = len(train_dataloader) / 5
    for epoch in range(start_epoch, epoch_sayisi):
        print(f"Epoch {epoch} başlıyor...")
        data_iter = iter(train_dataloader)

        # İlk epoch için başlangıç batch'ine kadar olan batch'leri atla
        if epoch == start_epoch:
            for _ in range(start_batch_index):
                next(data_iter)
            print(f"{start_batch_index} batch atlandı")
        for batch_index, batch in tqdm(enumerate(data_iter, start=start_batch_index),
                                       total=len(train_dataloader) - start_batch_index,
                                       desc=f"Processing {epoch}"):
            model.train()
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            save_checkpoint(model, optimizer, epoch, checkpoint_path, batch_index,
                            checkpoint_name,
                            val_accuracy_list, val_exact_match_list, val_f1_score, var_lcs_list,
                            batch_index % train_dataloader_len_5 == train_dataloader_len_5 - 1)
        val_accuracy, val_exact_match, val_f1_score, val_lcs = evaluate_model(model, validation_dataloader, device, tokenizer)
        print(f"\nEpoch {epoch} tamamlandı. F1: {val_f1_score_list}, Exact Match: {val_exact_match}")
        print(f"Accuracy: {val_accuracy}. LCS: {val_lcs}")
        val_accuracy_list.append(val_accuracy)
        val_exact_match_list.append(val_exact_match)
        val_f1_score_list.append(val_f1_score)
        var_lcs_list.append(val_lcs)
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path, -1, 
                        val_accuracy_list, val_exact_match_list, val_f1_score, var_lcs_list,
                        checkpoint_name, True)
        start_batch_index = 0
    plot_metrics(val_accuracy_list, lr, bs, epoch_sayisi, gorsel_yolu, "Accuracy")
    plot_metrics(val_exact_match_list, lr, bs, epoch_sayisi, gorsel_yolu, "Exact Match")
    plot_metrics(val_f1_score_list, lr, bs, epoch_sayisi, gorsel_yolu, "F1 Score")
    plot_metrics(var_lcs_list, lr, bs, epoch_sayisi, gorsel_yolu, "LCS")
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
    return generated_text.replace(prompt_text,"").strip()

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
    return generated_text.replace(prompt_text,"").strip()


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