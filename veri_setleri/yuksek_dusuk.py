import random
import pandas as pd
import os
DURUM = "Durum: "
BILGI = "Bilgi: "
yuksek_alternatifler = [
    "oldukça yüksek",
    "hayli yüksek",
    "çok yüksek"
]

orta_alternatifler = [
    "orta",
    "normal"
]
dusuk_alternatifler = [
    "oldukça düşük",
    "hayli düşük",
    "çok düşük"
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
        # Dosya yolu oluştur
        output_filepath = os.path.join('..', 'ana_veri_seti.csv')

        # Dosyayı yazmak için aç, başlık ekleyerek
        with open(output_filepath, 'w') as file:
            file.write("text\n")  # Sütun başlığı

            for index, row in self.df.iterrows():
                # Her satır için tanımlayıcı cümleleri oluştur
                cumleler = self.tanimlayici_cumle_olustur(row)
                for cumle in cumleler:
                    # Dosyaya yaz
                    file.write(f"{cumle}\n")
    def tanimlayici_cumle_olustur(self, row):
        duzenlenmis_cumleler_matrisi = []
        cumle_matrisi = self.cumle_olustur_fun(row)
        for cumleler in cumle_matrisi:
            tani_indexi = None
            for i, cumle in enumerate(cumleler):
                if BILGI in cumle:
                    tani_indexi = i
                    break
            # Tanıyı sona taşıma
            if tani_indexi is not None:
                cumleler.append(cumleler.pop(tani_indexi))
            # İlk cümleyi soru ifadesi ile değiştirme
            cumleler[0] = DURUM + cumleler[0]
            cumleler[-1] = BILGI + cumleler[-1]
            duzenlenmis_cumleler_matrisi.append(' '.join(cumleler))
        return duzenlenmis_cumleler_matrisi