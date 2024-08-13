import os
import csv
import json
from collections import defaultdict
import argparse


def process_csv_files(root_folder):
    merged_data = []
    skipped_rows = defaultdict(list)
    title_person_count = defaultdict(set)
    person_question_count = defaultdict(int)

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv") and file != "merged_sonuclar.csv":
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    consecutive_skipped = []
                    for row_num, row in enumerate(reader, start=1):
                        if all(row.values()):
                            merged_data.append(row)
                            title_person_count[row["unvan"]].add(row["isim"])
                            person_question_count[row["isim"]] += 1
                            if consecutive_skipped:
                                skipped_rows[file_path].extend(
                                    format_consecutive(consecutive_skipped)
                                )
                                consecutive_skipped = []
                        else:
                            consecutive_skipped.append(row_num)
                    if consecutive_skipped:
                        skipped_rows[file_path].extend(
                            format_consecutive(consecutive_skipped)
                        )

    # Write merged data to a new CSV file
    output_file = os.path.join(root_folder, "merged_sonuclar.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["isim", "unvan", "soru", "indis", "model_adi", "puan"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

    # Generate statistics
    total_elements = len(merged_data)
    unique_titles = defaultdict(int)
    unique_name_title_pairs = defaultdict(int)
    model_scores = defaultdict(list)

    for row in merged_data:
        unique_titles[row["unvan"]] += 1
        name_title_pair = (row["isim"], row["unvan"])
        unique_name_title_pairs[name_title_pair] += 1
        model_scores[row["model_adi"]].append(float(row["puan"]))

    model_averages = {
        model: sum(scores) / len(scores) for model, scores in model_scores.items()
    }

    # Create JSON output
    json_output = {
        "total_elements": total_elements,
        "unique_titles": dict(unique_titles),
        "unique_name_title_pairs": {
            f"{name}-{title}": count
            for (name, title), count in unique_name_title_pairs.items()
        },
        "total_unique_name_title_pairs": len(unique_name_title_pairs),
        "model_averages": model_averages,
        "skipped_rows": {k: v for k, v in skipped_rows.items() if v},
        "title_person_count": {
            title: len(persons) for title, persons in title_person_count.items()
        },
        "person_question_count": person_question_count,
    }

    # Write JSON output
    json_file = os.path.join(root_folder, "statistics.json")
    with open(json_file, "w", encoding="utf-8") as jsonfile:
        json.dump(json_output, jsonfile, ensure_ascii=False, indent=2)

    print(f"Merged CSV saved as: {output_file}")
    print(f"Statistics JSON saved as: {json_file}")


def format_consecutive(numbers):
    result = []
    start = numbers[0]
    prev = start
    for num in numbers[1:] + [None]:
        if num != prev + 1:
            if start == prev:
                result.append(str(start))
            else:
                result.append(f"{start}-{prev}")
            start = num
        prev = num
    return result


# Argüman ayrıştırıcıyı oluştur
parser = argparse.ArgumentParser(
    description="CSV dosyalarını işle ve istatistikleri çıkar."
)
parser.add_argument(
    "root_folder", type=str, help="İşlenecek CSV dosyalarının bulunduğu kök klasör"
)

# Argümanları ayrıştır
args = parser.parse_args()

# Fonksiyonu çağır
process_csv_files(args.root_folder)
