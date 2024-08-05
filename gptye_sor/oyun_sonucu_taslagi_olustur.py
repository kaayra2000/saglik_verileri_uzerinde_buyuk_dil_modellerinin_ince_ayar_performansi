import pandas as pd

# CSV dosyasını oku
input_file = "gpt_oyun_fiksturu_cevapli.csv"
output_file = "gpt_oyun_sonuclari.csv"

# Veriyi oku
df = pd.read_csv(input_file)
df2 = pd.read_csv("gpt_oyun_fiksturu.csv")
# Yeni DataFrame oluştur
new_df = pd.DataFrame(
    {
        "indis": range(len(df)),
        "oyuncu1": df2["model1"],
        "oyuncu2": df2["model2"],
        "mac_sonucu": -1,
        "veri_indisi": df["index"],
    }
)

# Yeni CSV dosyasını kaydet
new_df.to_csv(output_file, index=False)

print(f"Veriler '{output_file}' dosyasına kaydedildi.")
