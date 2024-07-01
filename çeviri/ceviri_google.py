from deep_translator import GoogleTranslator
from fonksiyonlar import translate_text_google, cevir_kaydet, veri_yolu_al

translator = GoogleTranslator(source="en", target="tr")

veri_yolu = veri_yolu_al()
cevir_kaydet(veri_yolu, translate_text_google, translator)
