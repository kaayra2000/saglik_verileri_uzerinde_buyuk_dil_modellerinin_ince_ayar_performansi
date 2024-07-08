import os
import json
import csv
import re
from collections import defaultdict
import argparse


def format_missing_indices(missing_indices):
    sorted_indices = sorted(missing_indices)
    ranges = []
    start = sorted_indices[0]
    end = sorted_indices[0]

    for i in sorted_indices[1:]:
        if i == end + 1:
            end = i
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = end = i

    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def get_numeric_part(filename):
    match = re.search(r"(\d+)-(\d+)_output", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def process_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    # Dosya adlarını sayısal olarak sıralama
    json_files.sort(key=lambda x: get_numeric_part(x)[0])
    csv_file_prefix = json_files[0].split("_")[0]
    combined_data = defaultdict(lambda: {"question_content": "", "question_answer": ""})

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data["data"]:
                custom_id = item["custom_id"]
                custom_id_parts = custom_id.split("-")
                index = int(custom_id_parts[1])
                content = item["response"]["body"]["choices"][0]["message"]["content"]

                if "question" in custom_id:
                    combined_data[index]["question_content"] = content
                elif "answer" in custom_id:
                    combined_data[index]["question_answer"] = content

    return combined_data, csv_file_prefix


def write_combined_csv(prefix, combined_data):
    csv_filename = f"../{prefix}.csv"
    cleaned_csv_filename = f"../{prefix}_cleaned.csv"

    # Mevcut cleaned csv dosyasını oku
    existing_cleaned_data = []
    if os.path.exists(cleaned_csv_filename):
        with open(cleaned_csv_filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_cleaned_data = list(reader)

    # Mevcut normal csv dosyasını oku
    existing_data = []
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

    # Yeni verilerle güncellenmiş cleaned csv dosyasını yaz
    with open(
        cleaned_csv_filename, "w", newline="", encoding="utf-8"
    ) as cleaned_csvfile:
        fieldnames = [
            "doctor_title",
            "doctor_speciality",
            "question_content",
            "question_answer",
        ]
        writer = csv.DictWriter(cleaned_csvfile, fieldnames=fieldnames)
        writer.writeheader()

        last_written_index = len(existing_cleaned_data)
        for i, row in enumerate(existing_cleaned_data):
            if i in combined_data:
                row["question_content"] = combined_data[i]["question_content"]
                row["question_answer"] = combined_data[i]["question_answer"]
            writer.writerow(row)

        for i, row in enumerate(existing_data, start=last_written_index):
            if i in combined_data:
                row["question_content"] = combined_data[i]["question_content"]
                row["question_answer"] = combined_data[i]["question_answer"]
                writer.writerow(row)
            else:
                break

    missing_indices = set(
        range(last_written_index + 1, max(combined_data.keys()) + 1)
    ) - set(combined_data.keys())
    if missing_indices:
        formatted_indices = format_missing_indices(missing_indices)
        print(
            f"Some indices were missing and could not be written: {formatted_indices}"
        )


def main(folder_path):
    combined_data, csv_file_prefix = process_json_files(folder_path)
    write_combined_csv(csv_file_prefix, combined_data)


if __name__ == "__main__":
    # folder path argümanını al
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Path to the folder containing JSON files")
    args = parser.parse_args()
    folder_path = args.folder_path
    main(folder_path)
