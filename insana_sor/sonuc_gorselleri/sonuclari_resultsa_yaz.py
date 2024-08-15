import json
import os


def update_result_files(root_folder, statistics_file):
    # JSON dosyasını oku
    with open(statistics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Her model için overall_average'ı al
    model_averages = data["Model Sonuçları"]["overall_average"]

    for model, stats in model_averages.items():
        model_folder = os.path.join(root_folder, model)

        # result_averaged.json ve result.json dosyalarını güncelle
        for filename in ["result_averaged.json", "result.json"]:
            file_path = os.path.join(model_folder, filename)

            if os.path.exists(file_path):
                # Mevcut dosyayı oku
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                # Yeni veriyi en başa ekle
                existing_data["insan_sonuclari"] = stats

                # Dosyayı güncelle
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

                print(f"Updated {file_path}")
            else:
                print(f"File not found: {file_path}")


# Kullanım
root_folder = "../../sonuclar"
statistics_file = "statistics.json"
update_result_files(root_folder, statistics_file)
