from openai import OpenAI
from fonksiyonlar import cevir_kaydet_json, translate_text_openai, veri_yolu_al

veri_yolu = veri_yolu_al()
with open("../api_key.txt", "r") as file:
    api_key = file.read().strip()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

cevir_kaydet_json(veri_yolu, translate_text_openai, client)
