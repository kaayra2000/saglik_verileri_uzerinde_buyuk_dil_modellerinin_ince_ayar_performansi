from deep_translator import GoogleTranslator
import sys

sys.path.append("..")
from fonksiyonlar import translate_text_google, cevir_kaydet_json, veri_yolu_al

translator = GoogleTranslator(source="en", target="tr")

veri_yolu = veri_yolu_al()
cevir_kaydet_json(veri_yolu, translate_text_google, translator)
