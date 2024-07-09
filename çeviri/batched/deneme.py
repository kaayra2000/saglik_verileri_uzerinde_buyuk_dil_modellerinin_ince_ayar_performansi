import csv
import random


def count_csv_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Header'ı atla
        rows = list(reader)
        row_count = len(rows)

        # Rastgele 3 eleman seç
        random_indices = random.sample(range(row_count), min(3, row_count))
        random_samples = [
            (index + 1, rows[index]) for index in random_indices
        ]  # Index +1 to account for the header

    return row_count, random_samples


# Kullanım
csv_file_path = "../icliniq_medical_qa_cleaned.csv"
num_rows, random_samples = count_csv_rows(csv_file_path)
print(f"CSV dosyasında {num_rows} eleman var.")
print("Rastgele 3 eleman:")
for index, sample in random_samples:
    print(f"Indis: {index}, Eleman: {sample}")
