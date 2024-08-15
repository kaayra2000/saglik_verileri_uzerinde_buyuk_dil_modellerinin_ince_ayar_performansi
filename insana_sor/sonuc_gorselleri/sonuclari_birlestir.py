import os
import csv
import json
from collections import defaultdict, Counter, OrderedDict
import argparse
import statistics


def calculate_statistics(scores):
    score_counts = Counter(scores)
    return {
        "average": statistics.mean(scores),
        "median": statistics.median(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "min": min(scores),
        "max": max(scores),
        "zero_count": score_counts[0],
        "positive_count": sum(
            count for score, count in score_counts.items() if score > 0
        ),
        "negative_count": sum(
            count for score, count in score_counts.items() if score < 0
        ),
        "score_distribution": OrderedDict(sorted(score_counts.items())),
    }


def process_csv_files(root_folder):
    merged_data = []
    skipped_rows = defaultdict(list)
    title_person_count = defaultdict(set)
    person_question_count = defaultdict(int)
    expert_question_count = 0
    non_expert_question_count = 0
    institution_question_count = defaultdict(int)
    expert_count = 0
    non_expert_count = 0

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
                            institution_question_count[row["kurum"]] += 1
                            if row["uzmanlık"].lower() != "yok":
                                expert_count += 1
                            else:
                                non_expert_count += 1
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
        fieldnames = [
            "isim",
            "unvan",
            "soru",
            "indis",
            "model_adi",
            "uzmanlık",
            "kurum",
            "puan",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

    # Generate statistics
    total_elements = len(merged_data)
    unique_titles = defaultdict(int)
    model_scores = defaultdict(list)
    title_model_scores = defaultdict(lambda: defaultdict(list))

    for row in merged_data:
        unique_titles[row["unvan"]] += 1
        name_title_pair = (row["isim"], row["unvan"])
        model_scores[row["model_adi"]].append(float(row["puan"]))
        title_model_scores[row["unvan"]][row["model_adi"]].append(float(row["puan"]))

    model_averages = {
        "overall_average": {
            model: calculate_statistics(scores)
            for model, scores in model_scores.items()
        },
        "specific_averages": {
            title: {
                model: calculate_statistics(scores)
                for model, scores in model_scores.items()
            }
            for title, model_scores in title_model_scores.items()
        },
    }

    # Create JSON output
    json_output = {
        "Toplam Cevap Sayısı": total_elements,
        "Ünvanlara Göre Cevap Sayısı": dict(unique_titles),
        "Ünvan-İnsan Sayısı": {
            title: len(persons) for title, persons in title_person_count.items()
        },
        "İnsan-Cevap Sayısı": person_question_count,
        "Uzmanlık Olan Doktor Sayısı": expert_count,
        "Uzman Olmayan Doktor Sayısı": non_expert_count,
        "Kurum - Cevap Sayısı": institution_question_count,
        "Farklı Kurum Sayısı": len(institution_question_count),
        "Model Sonuçları": model_averages,
        "skipped_rows": {k: v for k, v in skipped_rows.items() if v},
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
