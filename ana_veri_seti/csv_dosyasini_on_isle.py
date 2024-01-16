import pandas as pd
"""
    Veri setini işler ve metin dönüşümü yapar.
"""
veri_seti_adi = "diabetes_prediction_dataset.csv"
# veri setini oku ve data'ya at
data = pd.read_csv(veri_seti_adi)

# DataFrame oluşturma
df = pd.DataFrame(data)

# Metin dönüşüm fonksiyonu
def convert_to_text(row):
    hypertension_status = "has hypertension" if row["hypertension"] == 1 else "does not have hypertension"
    heart_disease_status = "has heart disease" if row["heart_disease"] == 1 else "does not have heart disease"
    diabetes_status = "has been diagnosed with diabetes" if row["diabetes"] == 1 else "has not been diagnosed with diabetes"

    return (f"The patient is a {row['age']} year old {row['gender']}. "
            f"The patient {hypertension_status}, {heart_disease_status}, "
            f"and {diabetes_status}. Smoking history: {row['smoking_history']}. "
            f"BMI: {row['bmi']}. HbA1c level: {row['HbA1c_level']}. "
            f"Blood glucose level: {row['blood_glucose_level']}.")

# Dönüştürülen metinleri yeni bir sütuna ekleme
df["text"] = df.apply(convert_to_text, axis=1)

# Dönüştürülen metinleri yeni bir dosyaya kaydetme
output_filename =  "islenmis_" + veri_seti_adi
df["text"].to_csv(output_filename, index=False)

