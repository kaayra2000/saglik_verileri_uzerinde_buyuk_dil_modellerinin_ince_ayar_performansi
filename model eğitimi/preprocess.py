import re
import nltk
from nltk.corpus import stopwords
from string import punctuation

# NLTK kütüphanesinin duraksama kelimelerini indir
nltk.download('stopwords')

# Türkçe duraksama kelimeleri (Eğer nltk'de Türkçe destekleniyorsa, aksi takdirde özel liste kullanılabilir)
try:
    stop_words = set(stopwords.words('turkish'))
except:
    stop_words = {"ve", "ama", "fakat", "çünkü", "ancak", "yani", "ise", "da", "de", "mi", "mı", "mu", "mü"}

# Türkçe ön işleme fonksiyonu
def preprocess_turkish_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    # Sayıları kaldır
    text = re.sub(r"\d+", "", text)
    # Fazladan boşlukları kaldır
    text = re.sub(r"\s+", " ", text).strip()
    # Duraksama kelimelerini kaldır
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text
