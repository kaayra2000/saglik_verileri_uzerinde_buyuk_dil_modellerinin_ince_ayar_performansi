import pandas as pd

# Dosyaları okuyalım
file1_path = "gpt_oyun_fiksturu.csv"
file2_path = "gptye_sorulacak_veri_cevapli.csv"

file1_df = pd.read_csv(file1_path)
file2_df = pd.read_csv(file2_path)

# Sonuçları saklayacak liste
results = []

# Eşleşmeleri saymak için bir sözlük
pair_counts = {tuple(pair): 0 for pair in file1_df.values}

# Her satırı gezerek model eşleşmelerini say
for index, fikstur_row in file1_df.iterrows():
    model1 = fikstur_row["model1"]
    model2 = fikstur_row["model2"]
    pair = (model1, model2)
    pair_count = pair_counts[pair]
    pair_counts[pair] += 1
    pair_counts[(model2, model1)] += 1
    row = file2_df.iloc[pair_count]
    # Her eşleşme sayısına ulaşıldığında veriyi topla
    result = {
        "index": row["index"],
        "question": row["question"],
        "model1": row[model1],
        "model2": row[model2],
    }
    results.append(result)

# Sonuçları bir DataFrame'e dönüştürüp CSV olarak kaydedelim
results_df = pd.DataFrame(results)
results_df.to_csv("sonuc.csv", index=False)

print("Sonuçlar başarıyla kaydedildi: sonuc.csv")
