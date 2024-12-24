import os
import csv
import json
from collections import defaultdict, Counter, OrderedDict
import argparse
import statistics
import pandas as pd
from scipy.stats import f_oneway
from pingouin import cronbach_alpha

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


def calculate_question_statistics(scores):
    if len(scores) > 1:
        return {"standart_sapma": statistics.stdev(scores), "sayı": len(scores)}
    else:
        return {"standart_sapma": 0, "sayı": len(scores)}


def calculate_model_question_stats(model_question_scores):
    model_question_stats = {}
    for model, questions in model_question_scores.items():
        model_question_stats[model] = {}
        for question, scores in questions.items():
            model_question_stats[model][question] = calculate_question_statistics(
                scores
            )
    return model_question_stats
# ANOVA ve Cronbach's Alpha hesaplamaları için fonksiyonlar
def calculate_anova(data, group_col, value_col):
    """
    ANOVA hesaplaması yapar.
    """
    groups = data.groupby(group_col)[value_col].apply(list)
    return f_oneway(*groups)

def calculate_cronbach_alpha(data):
    """
    Cronbach's Alpha hesaplaması yapar.
    """
    return cronbach_alpha(data)

def process_data(file_path):
    """
    Verileri işler ve hem soru hem de doktor bazında ANOVA ve Cronbach's Alpha hesaplamalarını yapar.
    """
    # Veriyi yükle
    df = pd.read_csv(file_path)

    # Soru bazında ANOVA hesaplaması
    anova_results_by_question = {}
    for question in df['soru'].unique():
        question_data = df[df['soru'] == question]
        anova_result = calculate_anova(question_data, 'isim', 'puan')
        anova_results_by_question[int(question)] = {
            'F-statistic': float(anova_result.statistic),
            'p-value': float(anova_result.pvalue)
        }

    # Doktor bazında ANOVA hesaplaması
    anova_results_by_doctor = {}
    for doctor in df['isim'].unique():
        doctor_data = df[df['isim'] == doctor]
        anova_result = calculate_anova(doctor_data, 'soru', 'puan')
        anova_results_by_doctor[doctor] = {
            'F-statistic': float(anova_result.statistic),
            'p-value': float(anova_result.pvalue)
        }

    # Soru bazında Cronbach's Alpha hesaplaması
    cronbach_results_by_question = {}
    for question in df['soru'].unique():
        question_data = df[df['soru'] == question].pivot(index='isim', columns='model_adi', values='puan')
        alpha, _ = calculate_cronbach_alpha(question_data)
        cronbach_results_by_question[int(question)] = float(alpha)

    # Doktor bazında Cronbach's Alpha hesaplaması
    cronbach_results_by_doctor = {}
    for doctor in df['isim'].unique():
        doctor_data = df[df['isim'] == doctor].pivot(index='soru', columns='model_adi', values='puan')
        alpha, _ = calculate_cronbach_alpha(doctor_data)
        cronbach_results_by_doctor[doctor] = float(alpha)

    return {
        'anova_by_question': anova_results_by_question,
        'anova_by_doctor': anova_results_by_doctor,
        'cronbach_by_question': cronbach_results_by_question,
        'cronbach_by_doctor': cronbach_results_by_doctor
    }

def process_csv_files(root_folder):
    merged_data = []
    skipped_rows = defaultdict(list)
    title_person_count = defaultdict(set)
    person_question_count = defaultdict(int)
    institution_question_count = defaultdict(int)
    model_question_scores = defaultdict(lambda: defaultdict(list))
    expert_count = 0
    non_expert_count = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv") and file != "merged_sonuclar.csv":
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                with open(file_path, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    consecutive_skipped = []
                    for row_num, row in enumerate(reader, start=1):
                        if all(
                            value for key, value in row.items() if key != "uzmanlık"
                        ):
                            merged_data.append(row)
                            title_person_count[row["unvan"]].add(row["isim"])
                            person_question_count[row["isim"]] += 1
                            institution_question_count[row["kurum"]] += 1
                            if (
                                row
                                and "uzmanlık" in row
                                and row["uzmanlık"]
                                and row["uzmanlık"].lower() != "yok"
                            ):
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
    anova_and_cronbach_results = process_data("merged_sonuclar.csv")
    for row in merged_data:
        model_question_scores[row["model_adi"]][row["soru"]].append(float(row["puan"]))
        unique_titles[row["unvan"]] += 1
        name_title_pair = (row["isim"], row["unvan"])
        model_scores[row["model_adi"]].append(float(row["puan"]))
        title_model_scores[row["unvan"]][row["model_adi"]].append(float(row["puan"]))
    # Model-soru çiftlerinin istatistiklerini hesapla
    model_question_stats = calculate_model_question_stats(model_question_scores)
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
        "Model-Soru Çiftleri Standart Sapma": model_question_stats,
        "Model Sonuçları": model_averages,
        "skipped_rows": {k: v for k, v in skipped_rows.items() if v},
    }
    json_output.update(anova_and_cronbach_results)
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
