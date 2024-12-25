import json
import matplotlib.pyplot as plt
import numpy as np

# Renk paleti
colors = [
    "#1ABC9C",  # Turkuaz
    "#F39C12",  # Sıcak turuncu
    "#8E44AD",  # Koyu mor
    "#34495E",  # Koyu gri-mavi
    "#D35400"   # Koyu turuncu
]

# JSON dosyasını yükleme
with open("statistics.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Model-Soru Çiftleri Standart Sapma grafikleri
def plot_model_question_std_dev(data):
    model_data = data["Model-Soru Çiftleri Standart Sapma"]
    for i, (model_name, questions) in enumerate(model_data.items()):
        x = list(questions.keys())
        y = [q["standart_sapma"] for q in questions.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(x, y, color=colors[0])
        plt.ylabel("Standart Sapma")
        plt.xlabel("Soru Sayısı")
        plt.tight_layout()
        plt.savefig(f"{model_name}_soru_standart_sapma.svg", format="svg")
        plt.close()

# Model Sonuçları (average, median, std_dev) bar chart
def plot_model_overall_results(data):
    overall_data = data["Model Sonuçları"]["overall_average"]
    models = list(overall_data.keys())
    averages = [overall_data[model]["average"] for model in models]
    medians = [overall_data[model]["median"] for model in models]
    std_devs = [overall_data[model]["std_dev"] for model in models]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, averages, width, label="Average", color=colors[0])
    plt.bar(x, medians, width, label="Median", color=colors[1])
    plt.bar(x + width, std_devs, width, label="Std Dev", color=colors[2])

    plt.xticks(x, models)
    plt.ylabel("Puanlar")
    plt.legend(loc="upper right", bbox_to_anchor=(1.08, 1.05))
    plt.tight_layout()
    plt.savefig("model_overall_results.svg", format="svg")
    plt.close()

# ANOVA by Question grafikleri
def plot_anova_by_question(data):
    anova_data = data["anova_by_question"]
    x = list(anova_data.keys())
    f_stats = [anova_data[q]["F-statistic"] for q in x]
    p_values = [anova_data[q]["p-value"] for q in x]

    x_indices = np.arange(len(x))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x_indices - width/2, f_stats, width, label="F-statistic", color=colors[0])
    plt.bar(x_indices + width/2, p_values, width, label="p-value", color=colors[1])
    plt.xticks(x_indices, x)
    plt.ylabel("Değerler")
    plt.xlabel("Soru Sayısı")
    plt.legend(loc="upper right", bbox_to_anchor=(1.04, 1.05))
    plt.tight_layout()
    plt.savefig("anova_by_question.svg", format="svg")
    plt.close()

# ANOVA by Doctor grafikleri
def plot_anova_by_doctor(data):
    anova_data = data["anova_by_doctor"]
    doctors = [f"Doktor {i+1}" for i in range(len(anova_data))]
    f_stats = [anova_data[doc]["F-statistic"] for doc in anova_data]
    p_values = [anova_data[doc]["p-value"] for doc in anova_data]

    x = np.arange(len(doctors))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, f_stats, width, label="F-statistic", color=colors[0])
    plt.bar(x + width/2, p_values, width, label="p-value", color=colors[1])

    plt.xticks(x, doctors, rotation=45)
    plt.ylabel("Değerler")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("anova_by_doctor.svg", format="svg")
    plt.close()

# Cronbach by Question grafikleri
def plot_cronbach_by_question(data):
    cronbach_data = data["cronbach_by_question"]
    x = list(cronbach_data.keys())
    y = list(cronbach_data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color=colors[0])
    plt.ylabel("Cronbach Alpha")
    plt.xlabel("Soru Sayısı")
    plt.tight_layout()
    plt.savefig("cronbach_by_question.svg", format="svg")
    plt.close()

# Cronbach by Doctor grafikleri
def plot_cronbach_by_doctor(data):
    cronbach_data = data["cronbach_by_doctor"]
    doctors = [f"Doktor {i+1}" for i in range(len(cronbach_data))]
    y = list(cronbach_data.values())

    plt.figure(figsize=(12, 6))
    plt.bar(doctors, y, color=colors[0])
    plt.ylabel("Cronbach Alpha")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cronbach_by_doctor.svg", format="svg")
    plt.close()

# Fonksiyonları çağırma
plot_model_question_std_dev(data)
plot_model_overall_results(data)
plot_anova_by_question(data)
plot_anova_by_doctor(data)
plot_cronbach_by_question(data)
plot_cronbach_by_doctor(data)