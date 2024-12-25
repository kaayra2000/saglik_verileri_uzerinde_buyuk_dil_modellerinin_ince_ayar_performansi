import os
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

# JSON dosyasını oku
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Çizim fonksiyonu
def plot_metrics(data, output_dir):

    # Tüm veri kümelerini ve modelleri al
    datasets = list(data.keys())
    models = list(next(iter(data.values())).keys())

    # Çizilecek metrikler ve Türkçe etiketleri
    metrics = {
        "bleu1": "BLEU-1 Doğruluğu",
        "bleu2": "BLEU-2 Doğruluğu",
        "bleu3": "BLEU-3 Doğruluğu",
        "bleu4": "BLEU-4 Doğruluğu",
        "bleu": "Genel BLEU Doğruluğu",  # Genel BLEU skoru etiketi
        "rouge1": "ROUGE-1 Skoru",
        "rouge2": "ROUGE-2 Skoru",
        "rougeL": "ROUGE-L Skoru",
        "rougeLsum": "ROUGE-Lsum Skoru",
        "meteor": "METEOR Skoru",
        "bertscore-avg_precision": "BERTScore Ortalama Doğruluk",
        "bertscore-avg_recall": "BERTScore Ortalama Geri Çağırma",
        "bertscore-avg_f1": "BERTScore Ortalama F1",
        "cer": "Karakter Hata Oranı (CER)",
        "wer": "Kelime Hata Oranı (WER)"
    }
    fontsize = 16
    rotation = 25

    # BLEU doğruluklarını ayrı ayrı çizmek için özel bir döngü
    for i in range(4):  # BLEU-1, BLEU-2, BLEU-3, BLEU-4
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.2  # Bar genişliği
        x = np.arange(len(models))  # Model sayısına göre x ekseni

        for j, dataset in enumerate(datasets):
            metric_values = []
            for model in models:
                # BLEU doğruluklarını al
                model_data = data[dataset].get(model, {})
                precisions = model_data.get("bleu", {}).get("precisions", [0, 0, 0, 0])
                value = precisions[i] if i < len(precisions) else 0
                metric_values.append(value)

            # Barları çiz
            ax.bar(x + j * bar_width, metric_values, bar_width, label=dataset, color=colors[j % len(colors)])

        ax.set_ylabel(f"BLEU-{i+1} Doğruluğu", fontsize=fontsize)
        ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(models, rotation=rotation, ha="right", fontsize=fontsize)
        ax.legend(title="Veri Kümeleri", loc="upper right", bbox_to_anchor=(1.25, 1.05))  # Sağ üst köşeye taşı
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # SVG olarak kaydet
        output_path = os.path.join(output_dir, f"bleu{i+1}_dogruluklari.svg")
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close()

    # Genel BLEU skorunu çizmek için ekleme
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.2  # Bar genişliği
    x = np.arange(len(models))  # Model sayısına göre x ekseni

    for j, dataset in enumerate(datasets):
        metric_values = []
        for model in models:
            # Genel BLEU skorunu hesapla (geometrik ortalama)
            model_data = data[dataset].get(model, {})
            precisions = model_data.get("bleu", {}).get("precisions", [0, 0, 0, 0])
            if precisions:
                # Geometrik ortalama hesaplama
                bleu_score = np.prod([p for p in precisions if p > 0]) ** (1 / len(precisions))
            else:
                bleu_score = 0
            metric_values.append(bleu_score)

        # Barları çiz
        ax.bar(x + j * bar_width, metric_values, bar_width, label=dataset, color=colors[j % len(colors)])

    ax.set_ylabel("Genel BLEU Doğruluğu", fontsize=fontsize)
    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(models, rotation=rotation, ha="right", fontsize=fontsize)
    ax.legend(title="Veri Kümeleri", loc="upper right", bbox_to_anchor=(1.25, 1.05))  # Sağ üst köşeye taşı
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # SVG olarak kaydet
    output_path = os.path.join(output_dir, "genel_bleu_dogruluklari.svg")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()

    for metric, label in metrics.items():
        if metric.startswith("bleu"):  # BLEU doğrulukları zaten çizildi
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.2  # Bar genişliği
        x = np.arange(len(models))  # Model sayısına göre x ekseni

        for i, dataset in enumerate(datasets):
            metric_values = []
            for model in models:
                # İlgili metrik değerini al
                model_data = data[dataset].get(model, {})
                if metric.startswith("rouge"):
                    value = model_data.get("rouge", {}).get(metric, 0)
                elif metric == "meteor":
                    value = model_data.get("meteor", {}).get("meteor", 0)
                elif metric.startswith("bertscore"):
                    bert_metric = metric.split("-")[-1]
                    value = model_data.get("bertscore", {}).get(bert_metric, 0)
                elif metric == "cer":
                    value = model_data.get("cer", 0)
                elif metric == "wer":
                    value = model_data.get("wer", 0)
                else:
                    value = 0
                metric_values.append(value)

            # Barları çiz
            ax.bar(x + i * bar_width, metric_values, bar_width, label=dataset, color=colors[i % len(colors)])

        ax.set_ylabel(label, fontsize=fontsize)
        ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(models, rotation=rotation, ha="right", fontsize=fontsize)
        ax.legend(title="Veri Kümeleri", loc="upper right", bbox_to_anchor=(1.25, 1.05))  # Sağ üst köşeye taşı
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # SVG olarak kaydet
        output_path = os.path.join(output_dir, f"{metric}_skorlari.svg")
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close()

# Ana fonksiyon
def main():
    json_file = "veri_kumesi_ozelinde_sonuclar.json"
    output_dir = "."

    try:
        data = load_json(json_file)
    except FileNotFoundError as e:
        print(e)
        return

    plot_metrics(data, output_dir)
    print(f"Tüm görselleştirmeler '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()