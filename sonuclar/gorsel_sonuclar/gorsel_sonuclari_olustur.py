import matplotlib.pyplot as plt
import numpy as np
import json, os

show_titles = False
show_x_labels = False
def plot_custom_bar_chart(
    matrix,
    x_labels,
    colors,
    data_labels,
    show_data_lables=True,
    title="Başlık",
    x_axis_label="X Ekseni",
    y_axis_label="Y Ekseni",
    legend_location="upper right",
    bar_width=0.2,
    show_values=True,
    file_path="chart.svg",
    fig_size=(12, 8),
    x_label_rotation=45,
    x_label_fontsize=10,
    y_label_fontsize=10,
    title_fontsize=12,
    x_title_fontsize=12,
    subplots_adjust_top=0.95,
    subplots_adjust_bottom=0.2,
    subplots_adjust_left=0.2,
    subplots_adjust_right=0.8,
    value_fontsize=10,
    y_tick_label_fontsize=10,
    float_len=3,
    y_axis_start=None,
):
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=fig_size)

    # Her bir veri türü için barları çiz
    for i, (data_label, color) in enumerate(zip(data_labels, colors)):
        scores = [row[i] for row in matrix]

        # Y ekseninin başlangıç değeri belirtilmişse, barların yüksekliğini ayarla
        if y_axis_start is not None:
            bar_heights = [score - y_axis_start for score in scores]
        else:
            bar_heights = scores

        bars = ax.bar(
            x + i * bar_width,
            bar_heights,
            bar_width,
            label=data_label,
            color=color,
            bottom=y_axis_start,
        )

        # Her barın üstüne sayısal değerleri yaz
        if show_values:
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.annotate(
                    "{}".format(
                        round(score, float_len) if float_len > 0 else int(score)
                    ),
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        height + y_axis_start if y_axis_start is not None else height,
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=value_fontsize,
                )

    # Eksen ve etiket ayarları
    if show_x_labels:
        ax.set_xlabel(x_axis_label, fontsize=x_title_fontsize)
    ax.set_ylabel(y_axis_label, fontsize=y_label_fontsize)
    if show_titles:
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(
        x_labels, rotation=x_label_rotation, ha="right", fontsize=x_label_fontsize
    )
    ax.tick_params(axis="y", labelsize=y_tick_label_fontsize)

    # Y ekseni başlangıç değerini ayarla
    if y_axis_start is not None:
        ax.set_ylim(bottom=y_axis_start)

    if show_data_lables:
        ax.legend(
            data_labels,
            loc=legend_location,
            bbox_to_anchor=(1, 1),
        )

    plt.subplots_adjust(
        top=subplots_adjust_top,
        bottom=subplots_adjust_bottom,
        left=subplots_adjust_left,
        right=subplots_adjust_right,
    )

    # Grafik SVG olarak kaydedilmesi
    plt.savefig(file_path, format="svg")
def plot_training_losses(training_losses, model_names, colors):
    plt.figure(figsize=(12, 8))
    
    for i, model_name in enumerate(model_names):
        epochs, losses, _, _ = zip(*training_losses[i])
        plt.plot(epochs, losses, label=model_name, color=colors[i], marker='o')
    
    plt.xlabel('Döngü')
    plt.ylabel('Eğitim Kayıp Değeri')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('training_loss_comparison.svg')
    plt.close()

def plot_validation_losses(validation_losses, model_names, colors):
    plt.figure(figsize=(12, 8))
    
    for i, model_name in enumerate(model_names):
        if validation_losses[i]:  # Eğer doğrulama kaybı varsa
            epochs, losses = zip(*validation_losses[i])
            plt.plot(epochs, losses, label=model_name, color=colors[i], marker='s')
    
    plt.xlabel('Döngü')
    plt.ylabel('Validasyon Kayıp Değeri')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('validation_loss_comparison.svg')
    plt.close()

def plot_grad_norm(training_losses, model_names, colors):
    plt.figure(figsize=(12, 8))
    
    for i, model_name in enumerate(model_names):
        epochs, _, grad_norms, _ = zip(*training_losses[i])
        plt.plot(epochs, grad_norms, label=model_name, color=colors[i], marker='o')
    
    plt.xlabel('Döngü')
    plt.ylabel('Gradient Norm Değeri')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('grad_norm_comparison.svg')
    plt.close()

def plot_learning_rate(training_losses, model_names, colors):
    plt.figure(figsize=(12, 8))
    min_lr = float('inf')
    for i, model_name in enumerate(model_names):
        epochs, _, _, learning_rates = zip(*training_losses[i])
        plt.plot(epochs, learning_rates, label=model_name, color=colors[i], marker='o')
        min_lr = min(min_lr, min(learning_rates))
    plt.xlabel('Döngü')
    plt.ylabel('Öğrenme Oranı')
    plt.legend()
    plt.grid(True)
    
    # Y eksenini en düşük öğrenme oranının biraz altından başlat
    plt.ylim(bottom=min_lr * 0.9)
    
    plt.savefig('learning_rate_comparison.svg')
    plt.close()




def list_and_read_json_files(base_dir=".."):
    """
    Verilen dizindeki (base_dir) ve alt dizinlerindeki (2 derinlikteki) tüm dosyaları listeler ve result.json dosyalarını okur.
    """
    all_files = []
    json_contents = []

    # İki derinlikteki tüm dosyaları listele
    for root, dirs, files in os.walk(base_dir):
        if (
            root.count(os.sep) - base_dir.count(os.sep) < 3
        ):  # Sadece iki derinlik kontrolü
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)

                # result.json dosyalarını oku
                if file == "result_averaged.json":
                    with open(full_path, "r") as json_file:
                        try:
                            json_contents.append(json.load(json_file))
                            full_dir = os.path.dirname(full_path)
                            training_log_path = os.path.join(full_dir, "training_log.json")
                            with open(training_log_path, "r") as training_log_file:
                                training_log = json.load(training_log_file)
                                last_item = json_contents[-1]
                                last_key = list(last_item.keys())[0]
                                last_item[last_key]["training_log"] = training_log
                        except json.JSONDecodeError as e:
                            print(f"Error reading {full_path}: {e}")

    return all_files, json_contents



# İşlevi çalıştır ve sonuçları yazdır
all_files, json_contents = list_and_read_json_files()


colors = [
    "#1ABC9C",  # Turkuaz
    "#F39C12",  # Sıcak turuncu
    "#8E44AD",  # Koyu mor
    "#34495E",  # Koyu gri-mavi
    "#D35400"   # Koyu turuncu
]

model_names = [
    "Meta-Llama-3-8B",
    "SambaLingo-Turkish-Chat",
    "Trendyol-LLM-7b-chat-v1.8",
    "Turkish-Llama-8b-v0.1",
]

""""
BERT SKOR KISMI
"""
bert_labels = ["Ort. Precision", "Ort. Recall", "Ort. F1"]

bert_general_file_path = "bert_general_chart.svg"
bert_general_matrix = []

bert_specific_file_path = "bert_specific_chart.svg"
bert_spesific_model_file_path = "bert_specific_model_chart.svg"
bert_specific_matrix_model = []

for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            bert_specific_matrix_model.append(
                [
                    model_content["bertscore"]["avg_precision"],
                    model_content["bertscore"]["avg_recall"],
                    model_content["bertscore"]["avg_f1"],
                ]
            )
            bert_general_matrix.append(
                [
                    model_content["bertscore"]["avg_f1"],
                ]
            )
            break

# NumPy array'e dönüştürme
bert_specific_matrix = np.array(bert_specific_matrix_model)

# Matrisin transpozesini alma
bert_specific_matrix = np.transpose(bert_specific_matrix)
plot_custom_bar_chart(
    bert_specific_matrix_model,
    model_names,
    colors,
    bert_labels,
    show_data_lables=True,
    title="Farklı Modellerin BERTScore Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="BERTScore Değerleri",
    legend_location="upper right",
    bar_width=0.28,
    show_values=True,
    file_path=bert_spesific_model_file_path,
    fig_size=(12, 15),
    x_label_fontsize=17,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
    subplots_adjust_right=0.95,
    value_fontsize=12,
    y_tick_label_fontsize=17,
)

plot_custom_bar_chart(
    bert_general_matrix,
    model_names,
    colors,
    [bert_labels[2]],
    show_data_lables=False,
    title="Modellerin BERTScore F1 Değerleri",
    x_axis_label="Model",
    y_axis_label="BERTScore F1 Değerleri",
    legend_location="upper right",
    bar_width=0.5,
    show_values=True,
    file_path=bert_general_file_path,
    fig_size=(12, 15),
    x_label_fontsize=17,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.1,
    subplots_adjust_right=0.95,
    value_fontsize=14,
    y_tick_label_fontsize=17,
)

plot_custom_bar_chart(
    bert_specific_matrix,
    bert_labels,
    colors,
    model_names,
    show_data_lables=True,
    title="Farklı Modellerin BERTScore Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="BERTScore Değerleri",
    legend_location="upper left",
    bar_width=0.2,
    show_values=True,
    file_path=bert_specific_file_path,
    fig_size=(16, 10),
    x_label_fontsize=17,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.17,
    subplots_adjust_top=0.9,
    subplots_adjust_left=0.1,
    subplots_adjust_right=0.78,
    y_tick_label_fontsize=18,
    value_fontsize=12,
)

""""
BLEU KISMI
"""

bleu_labels = ["BLEU-1 Skoru", "BLEU-2 Skoru", "BLEU-3 Skoru", "BLEU-4 Skoru"]

bleu_general_label = ["Genel BLEU Skoru"]
bleu_general_file_path = "bleu_general_chart.svg"
bleu_general_matrix = []

bleu_specific_file_path = "bleu_specific_chart.svg"
bleu_spesific_model_file_path = "bleu_specific_model_chart.svg"
bleu_specific_matrix_model = []

for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            bleu_specific_matrix_model.append(model_content["bleu"]["precisions"])
            bleu_general_matrix.append([model_content["bleu"]["bleu"]])
            break

# NumPy array'e dönüştürme
bleu_specific_matrix = np.array(bleu_specific_matrix_model)

# Matrisin transpozesini alma
bleu_specific_matrix = np.transpose(bleu_specific_matrix)

plot_custom_bar_chart(
    bleu_specific_matrix_model,
    model_names,
    colors,
    bleu_labels,
    show_data_lables=True,
    title="Farklı Modellerin BLEU Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="BLEU Skor Değerleri",
    legend_location="upper right",
    bar_width=0.23,
    show_values=True,
    file_path=bleu_spesific_model_file_path,
    fig_size=(12, 10),
    x_label_fontsize=12,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
    subplots_adjust_right=0.95,
    value_fontsize=8,
    y_tick_label_fontsize=15,
    float_len=5,
)


plot_custom_bar_chart(
    bleu_general_matrix,
    model_names,
    colors,
    bleu_general_label,
    show_data_lables=False,
    title="Farklı Modellerin Genel BLEU Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="Genel BLEU Skor Değerleri",
    legend_location="upper right",
    bar_width=0.4,
    show_values=True,
    file_path=bleu_general_file_path,
    fig_size=(14, 10),
    x_label_fontsize=12.2,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
    subplots_adjust_right=0.95,
    value_fontsize=15,
    y_tick_label_fontsize=15,
    float_len=5,
)


plot_custom_bar_chart(
    bleu_specific_matrix,
    bleu_labels,
    colors,
    model_names,
    show_data_lables=True,
    title="Farklı Modellerin BLEU Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="BLEU Skor Değerleri",
    legend_location="upper right",
    bar_width=0.23,
    show_values=True,
    file_path=bleu_specific_file_path,
    fig_size=(12, 10),
    x_label_fontsize=12,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.14,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
    subplots_adjust_right=0.95,
    value_fontsize=8,
    y_tick_label_fontsize=15,
    float_len=5,
)


""""
ROUGE KISMI
"""

rouge_labels = ["ROUGE-1 Skoru", "ROUGE-2 Skoru", "ROUGE-L Skoru", "ROUGE-Lsum Skoru"]

rouge_general_label = ["ROUGE-L Skoru"]
rouge_general_file_path = "rouge_general_chart.svg"
rouge_general_matrix = []

rouge_specific_file_path = "rouge_specific_chart.svg"
rouge_spesific_model_file_path = "rouge_specific_model_chart.svg"
rouge_specific_matrix_model = []
for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            rouge_specific_matrix_model.append(
                [
                    model_content["rouge"]["rouge1"],
                    model_content["rouge"]["rouge2"],
                    model_content["rouge"]["rougeL"],
                    model_content["rouge"]["rougeLsum"],
                ]
            )
            rouge_general_matrix.append([model_content["rouge"]["rougeL"]])
            break

# NumPy array'e dönüştürme
rouge_specific_matrix = np.array(rouge_specific_matrix_model)

# Matrisin transpozesini alma
rouge_specific_matrix = np.transpose(rouge_specific_matrix)

plot_custom_bar_chart(
    rouge_specific_matrix_model,
    model_names,
    colors,
    rouge_labels,
    show_data_lables=True,
    title="Farklı Modellerin ROUGE Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ROUGE Skor Değerleri",
    legend_location="upper right",
    bar_width=0.20,
    show_values=True,
    file_path=rouge_spesific_model_file_path,
    fig_size=(12, 10),
    x_label_fontsize=20,
    x_label_rotation=35,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.25,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.12,
    subplots_adjust_right=0.95,
    value_fontsize=8,
    y_tick_label_fontsize=15,
    float_len=4,
)


plot_custom_bar_chart(
    rouge_general_matrix,
    model_names,
    colors,
    rouge_general_label,
    show_data_lables=False,
    title="Farklı Modellerin ROUGE-L Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ROUGE-L Skor Değerleri",
    legend_location="upper right",
    bar_width=0.4,
    show_values=True,
    file_path=rouge_general_file_path,
    fig_size=(14, 11),
    x_label_fontsize=13,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.15,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.11,
    subplots_adjust_right=0.95,
    value_fontsize=15,
    y_tick_label_fontsize=15,
    float_len=4,
)


plot_custom_bar_chart(
    rouge_specific_matrix,
    rouge_labels,
    colors,
    model_names,
    show_data_lables=True,
    title="Farklı Modellerin ROUGE Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ROUGE Skor Değerleri",
    legend_location="upper right",
    bar_width=0.20,
    show_values=True,
    file_path=rouge_specific_file_path,
    fig_size=(12, 10),
    x_label_fontsize=12,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.14,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
    subplots_adjust_right=0.95,
    value_fontsize=8,
    y_tick_label_fontsize=15,
    float_len=4,
)


""""
METEOR KISMI
"""

meteor_general_label = ["METEOR Skoru"]
meteor_general_file_path = "meteor_general_chart.svg"
meteor_general_matrix = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            meteor_general_matrix.append([model_content["meteor"]["meteor"]])
            break


plot_custom_bar_chart(
    meteor_general_matrix,
    model_names,
    colors,
    meteor_general_label,
    show_data_lables=False,
    title="Farklı Modellerin METEOR Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="METEOR Skor Değerleri",
    legend_location="upper right",
    bar_width=0.4,
    show_values=True,
    file_path=meteor_general_file_path,
    fig_size=(14, 11),
    x_label_fontsize=13,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.15,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.11,
    subplots_adjust_right=0.95,
    value_fontsize=15,
    y_tick_label_fontsize=15,
    float_len=4,
)


""""
CER-WER KISMI
"""

cer_wer_general_label = ["CER Skoru", "WER Skoru"]
cer_wer_general_file_path = "cer_wer_general_chart.svg"
cer_wer_general_matrix = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            cer_wer_general_matrix.append([model_content["cer"], model_content["wer"]])
            break


plot_custom_bar_chart(
    cer_wer_general_matrix,
    model_names,
    colors,
    cer_wer_general_label,
    show_data_lables=True,
    title="Farklı Modellerin CER-WER Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="CER-WER Skor Değerleri",
    legend_location="upper left",
    bar_width=0.4,
    show_values=True,
    file_path=cer_wer_general_file_path,
    fig_size=(16, 11),
    x_label_fontsize=15,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.25,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.08,
    subplots_adjust_right=0.9,
    value_fontsize=15,
    y_tick_label_fontsize=15,
    float_len=2,
)

"""
ELO KISMI
"""

elo_general_label = [
    "GPT-4o",
    "LLama-3.1-70B-Instruct",
    "Copilot",
    "gemini-1.5-pro-001",
    "Claude 3.5 Sonnet",
]
elo_general_model_file_path = "elo_general_model_chart.svg"
elo_general_file_path = "elo_general_chart.svg"
elo_general_matrix_model = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            elo_general_matrix_model.append(
                [
                    model_content["ELO"]["gpt4o"],
                    model_content["ELO"]["llama3"],
                    model_content["ELO"]["copilot"],
                    model_content["ELO"]["gemini"],
                    model_content["ELO"]["claude"],
                ]
            )
            break

# NumPy array'e dönüştürme
elo_general_matrix = np.array(elo_general_matrix_model)

# Matrisin transpozesini alma
elo_general_matrix = np.transpose(elo_general_matrix)
plot_custom_bar_chart(
    elo_general_matrix_model,
    model_names,
    colors,
    elo_general_label,
    show_data_lables=True,
    title="Farklı Modellerin ELO Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ELO Skor Değerleri",
    legend_location="upper right",
    bar_width=0.15,
    show_values=True,
    file_path=elo_general_model_file_path,
    fig_size=(16, 11),
    x_label_fontsize=25,
    x_label_rotation=35,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.30,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.1,
    subplots_adjust_right=0.95,
    value_fontsize=9,
    y_tick_label_fontsize=15,
    float_len=0,
)

plot_custom_bar_chart(
    elo_general_matrix,
    elo_general_label,
    colors,
    model_names,
    show_data_lables=True,
    title="Farklı Modellerin ELO Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ELO Skor Değerleri",
    legend_location="upper left",
    bar_width=0.2,
    show_values=True,
    file_path=elo_general_file_path,
    fig_size=(16, 11),
    x_label_fontsize=15,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.25,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.08,
    subplots_adjust_right=0.78,
    value_fontsize=10,
    y_tick_label_fontsize=15,
    float_len=0,
)


"""
WinPct KISMI
"""

winpct_wer_general_label = [
    "GPT-4o",
    "LLama-3.1-70B-Instruct",
    "Copilot",
    "gemini-1.5-pro-001",
    "Claude 3.5 Sonnet",
]
winpct_general_model_file_path = "winpct_general_model_chart.svg"
winpct_general_file_path = "winpct_general_chart.svg"
winpct_general_matrix_model = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            winpct_general_matrix_model.append(
                [
                    model_content["WinPct"]["gpt4o"],
                    model_content["WinPct"]["llama3"],
                    model_content["WinPct"]["copilot"],
                    model_content["WinPct"]["gemini"],
                    model_content["WinPct"]["claude"],
                ]
            )
            break
# NumPy array'e dönüştürme
winpct_general_matrix = np.array(winpct_general_matrix_model)

# Matrisin transpozesini alma
winpct_general_matrix = np.transpose(winpct_general_matrix)

plot_custom_bar_chart(
    winpct_general_matrix_model,
    model_names,
    colors,
    winpct_wer_general_label,
    show_data_lables=True,
    title="Farklı Modellerin WinPct Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="WinPct Skor Değerleri",
    legend_location="upper right",
    bar_width=0.15,
    show_values=True,
    file_path=winpct_general_model_file_path,
    fig_size=(16, 11),
    x_label_fontsize=23,
    x_label_rotation=35,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.25,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.1,
    subplots_adjust_right=0.96,
    value_fontsize=11.5,
    y_tick_label_fontsize=15,
    float_len=2,
)


plot_custom_bar_chart(
    winpct_general_matrix,
    winpct_wer_general_label,
    colors,
    model_names,
    show_data_lables=True,
    title="Farklı Modellerin WinPct Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="WinPct Skor Değerleri",
    legend_location="upper right",
    bar_width=0.2,
    show_values=False,
    file_path=winpct_general_file_path,
    fig_size=(16, 11),
    x_label_fontsize=15,
    x_label_rotation=25,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.17,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.08,
    subplots_adjust_right=0.96,
    value_fontsize=13,
    y_tick_label_fontsize=15,
    float_len=1,
)

"""
İnsan sonuçları kısmı
"""
insan_ortalamalari_general_matrix = []
insan_ortalamalari_label = ["İnsan Ortalamaları"]
insan_ortalamalari_general_file_path = "insan_ortalamalari_general_chart.svg"
for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            insan_ortalamalari_general_matrix.append(
                [
                    model_content["insan_sonuclari"]["average"],
                ]
            )
            break


plot_custom_bar_chart(
    insan_ortalamalari_general_matrix,
    model_names,
    colors,
    insan_ortalamalari_label,
    show_data_lables=False,
    title="Doktor Değerlendirmeleri",
    x_axis_label="Model",
    y_axis_label="Ortalama Doktor Puanı",
    legend_location="upper left",
    bar_width=0.4,
    show_values=True,
    file_path=insan_ortalamalari_general_file_path,
    fig_size=(16, 11),
    x_label_fontsize=25,
    x_label_rotation=35,
    y_label_fontsize=25,
    title_fontsize=28,
    x_title_fontsize=25,
    subplots_adjust_bottom=0.27,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.1,
    subplots_adjust_right=0.95,
    value_fontsize=15,
    y_tick_label_fontsize=15,
    float_len=2,
    y_axis_start=-10,
)


"""
Eğitim sonuçları kısmı
"""

training_losses = []
validation_losses = []
for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            training_log_content = content[model_name]["training_log"]
            training_losses.append([])
            validation_losses.append([])
            for entry in training_log_content:
                if "loss" in entry:
                    training_losses[-1].append((entry["epoch"], entry["loss"], entry["grad_norm"], entry["learning_rate"]))
                elif "eval_loss" in entry:
                    validation_losses[-1].append((entry["epoch"], entry["eval_loss"]))
            break
plot_training_losses(training_losses, model_names, colors)
plot_validation_losses(validation_losses, model_names, colors)
plot_grad_norm(training_losses, model_names, colors)
plot_learning_rate(training_losses, model_names, colors)