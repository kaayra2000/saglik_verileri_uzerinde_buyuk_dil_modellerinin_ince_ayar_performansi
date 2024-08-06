import matplotlib.pyplot as plt
import numpy as np
import json, os


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
    y_label_fontsize=10,  # y ekseni etiketi yazı tipi boyutu
    title_fontsize=12,  # Başlık yazı tipi boyutu
    x_title_fontsize=12,
    subplots_adjust_top=0.95,
    subplots_adjust_bottom=0.2,
    subplots_adjust_left=0.2,
    subplots_adjust_right=0.8,
    value_fontsize=10,  # Bar üstündeki sayıların yazı tipi boyutu
    y_tick_label_fontsize=10,  # Y ekseni etiketlerinin yazı tipi boyutu
    float_len=3,
):
    """
    Özelleştirilebilir bar chart çizen ve SVG olarak kaydeden fonksiyon.

    Parametreler:
    - matrix: Matris verisi (satır sayısı model sayısı kadar, sütun sayısı veri türü kadar).
    - x_labels: X ekseni başlıkları (satır sayısı kadar).
    - colors: Renk listesi (satır sayısı kadar).
    - data_labels: Veri türü başlıkları (sütun sayısı kadar).
    - title: Grafik başlığı (varsayılan: 'Başlık').
    - x_axis_label: X ekseni başlığı (varsayılan: 'X Ekseni').
    - y_axis_label: Y ekseni başlığı (varsayılan: 'Y Ekseni').
    - legend_location: Sağ üstteki açıklamanın konumu (varsayılan: 'center left').
    - bar_width: Barların kalınlığı (varsayılan: 0.2).
    - show_values: Barların üstünde sayısal değerleri gösterme (varsayılan: True).
    - file_path: SVG dosyasının kaydedileceği dosya yolu (varsayılan: 'chart.svg').
    - fig_size: Grafik boyutu (varsayılan: (12, 8)).
    - x_label_rotation: X ekseni etiketlerinin dönüş açısı (varsayılan: 45 derece).
    - x_label_fontsize: X ekseni etiketlerinin yazı tipi boyutu (varsayılan: 10).
    - y_label_fontsize: Y ekseni etiketlerinin yazı tipi boyutu (varsayılan: 10).
    - title_fontsize: Başlık yazı tipi boyutu (varsayılan: 12).
    - value_fontsize: Bar üstündeki sayıların yazı tipi boyutu (varsayılan: 10).
    - y_tick_label_fontsize: Y ekseni etiketlerinin yazı tipi boyutu (varsayılan: 10).
    """

    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=fig_size)

    # Her bir veri türü için barları çiz
    for i, (data_label, color) in enumerate(zip(data_labels, colors)):
        scores = [row[i] for row in matrix]
        bars = ax.bar(
            x + i * bar_width, scores, bar_width, label=data_label, color=color
        )

        # Her barın üstüne sayısal değerleri yaz
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    "{}".format(round(height, float_len)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=value_fontsize,
                )

    # Eksen ve etiket ayarları
    ax.set_xlabel(x_axis_label, fontsize=x_title_fontsize)
    ax.set_ylabel(y_axis_label, fontsize=y_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(
        x_labels, rotation=x_label_rotation, ha="right", fontsize=x_label_fontsize
    )
    ax.tick_params(axis="y", labelsize=y_tick_label_fontsize)

    if show_data_lables:
        # Legend'i sağa ve dışa taşı
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
                        except json.JSONDecodeError as e:
                            print(f"Error reading {full_path}: {e}")

    return all_files, json_contents


# İşlevi çalıştır ve sonuçları yazdır
all_files, json_contents = list_and_read_json_files()


colors = ["#000080", "#0000ff", "#1e90ff", "#87ceeb"]

model_names = [
    "doktor-meta-llama-3-8b",
    "doktor-LLama2-sambanovasystems-7b",
    "doktor-Mistral-trendyol-7b",
    "doktor-llama-3-cosmos-8b",
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
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
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
    subplots_adjust_bottom=0.2,
    subplots_adjust_top=0.95,
    subplots_adjust_left=0.15,
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

elo_wer_general_label = ["ELO Skoru"]
elo_wer_general_file_path = "elo_general_chart.svg"
elo_wer_general_matrix = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            elo_wer_general_matrix.append([model_content["ELO"]])
            break


plot_custom_bar_chart(
    elo_wer_general_matrix,
    model_names,
    colors,
    elo_wer_general_label,
    show_data_lables=True,
    title="Farklı Modellerin ELO Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="ELO Skor Değerleri",
    legend_location="upper left",
    bar_width=0.4,
    show_values=True,
    file_path=elo_wer_general_file_path,
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
WinPct KISMI
"""

winpct_wer_general_label = ["WinPct Skoru"]
winpct_wer_general_file_path = "winpct_general_chart.svg"
winpct_wer_general_matrix = []


for model_name in model_names:
    for content in json_contents:
        if model_name in content:
            model_content = content[model_name]
            winpct_wer_general_matrix.append([model_content["WinPct"]])
            break


plot_custom_bar_chart(
    winpct_wer_general_matrix,
    model_names,
    colors,
    winpct_wer_general_label,
    show_data_lables=True,
    title="Farklı Modellerin WinPct Skor Performans Değerleri",
    x_axis_label="Model",
    y_axis_label="WinPct Skor Değerleri",
    legend_location="upper left",
    bar_width=0.4,
    show_values=True,
    file_path=winpct_wer_general_file_path,
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
