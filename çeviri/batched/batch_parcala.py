import json
import sys
import os

sys.path.append("..")
from fonksiyonlar import veri_yolu_al


def split_jsonl(file_path, chunk_size):
    with open(file_path, "r") as file:
        lines = file.readlines()

    total_lines = len(lines)
    chunks = [lines[i : i + chunk_size] for i in range(0, total_lines, chunk_size)]
    file_name = os.path.basename(file_path)
    splitted_files_directory = file_name.replace(".jsonl", "_splitted_files")
    if os.path.exists(splitted_files_directory):
        for file in os.listdir(splitted_files_directory):
            os.remove(os.path.join(splitted_files_directory, file))
    else:
        os.makedirs(splitted_files_directory, exist_ok=True)
    for i, chunk in enumerate(chunks):
        first_item = json.loads(chunk[0])
        last_item = json.loads(chunk[-1])

        first_id = first_item["custom_id"].split("-")[1]
        last_id = last_item["custom_id"].split("-")[1]

        output_file_name_postfix = f"{first_id}-{last_id}.jsonl"
        output_file_name = file_name.replace(".jsonl", f"_{output_file_name_postfix}")
        output_file_path = os.path.join(splitted_files_directory, output_file_name)
        with open(output_file_path, "w") as output_file:
            for line in chunk:
                output_file.write(line)

        print(f"Created file: {output_file_path}")


veri_yolu = veri_yolu_al()

# Kullanım
split_jsonl(veri_yolu, 1000)
