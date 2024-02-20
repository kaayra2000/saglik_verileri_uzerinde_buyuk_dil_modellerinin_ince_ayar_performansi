import random
import pandas as pd
import os
CEVAP = "Cevap: "
SORU = "Soru: "
yuksek_alternatifler = [
    "oldukça yüksek",
    "hayli yüksek",
    "çok yüksek",
    "anormal derecede yüksek",
    "aşırı yüksek",
    "kayda değer şekilde yüksek",
    "belirgin derecede yüksek"
]

orta_alternatifler = [
    "orta",
    "normal",
    "standart",
    "ortalama",
    "makul",
    "dengeli"
]
dusuk_alternatifler = [
    "oldukça düşük",
    "hayli düşük",
    "çok düşük",
    "anormal derecede düşük",
    "aşırı düşük",
    "kayda değer şekilde düşük",
    "belirgin derecede düşük"
]
# Rastgele cümle seçme fonksiyonu
def rastgele_cumle_sec(secenekler):
    return random.choice(secenekler)
# ön işleme işlemleri daha kolay olsun diye sınıf
class OnIsle():
    def __init__(self, veri_seti_adi, tanimlayici_cumle_olustur):
        self.veri_seti_adi = veri_seti_adi
        self.cumle_olustur_fun = tanimlayici_cumle_olustur
        self.init()
    def init(self):
        self.df = pd.read_csv(self.veri_seti_adi)
    def isle_kaydet(self):
        # Her satır için tanımlayıcı cümleler oluştur
        self.df['text'] = self.df.apply(self.tanimlayici_cumle_olustur, axis=1)
        selected_columns = self.df["text"]
        selected_columns = selected_columns.explode('text')
        # Sonucu yeni bir dosyaya kaydet
        output_filepath = os.path.join('..', 'labelsiz_' + self.veri_seti_adi)
        selected_columns.to_csv(output_filepath, index=False)
    def tanimlayici_cumle_olustur(self, row):
        duzenlenmis_cumleler_matrisi = []
        cumle_matrisi = self.cumle_olustur_fun(row)
        for cumleler in cumle_matrisi:
            tani_indexi = None
            for i, cumle in enumerate(cumleler):
                if CEVAP in cumle:
                    tani_indexi = i
                    break
            # Tanıyı sona taşıma
            if tani_indexi is not None:
                cumleler.append(cumleler.pop(tani_indexi))
            # İlk cümleyi soru ifadesi ile değiştirme
            cumleler[0] = SORU + cumleler[0]
            cumleler[-1] = CEVAP + cumleler[-1]
            duzenlenmis_cumleler_matrisi.append(' '.join(cumleler))
        return duzenlenmis_cumleler_matrisi