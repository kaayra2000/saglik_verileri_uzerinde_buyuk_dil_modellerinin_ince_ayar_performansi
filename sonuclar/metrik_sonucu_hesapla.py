import pandas as pd
import argparse
import evaluate
import json

# Metrikleri yükleme
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")


# compute_metrics fonksiyonunu tanımlama
def compute_metrics(predictions, references):
    print("Metrikler hesaplanıyor...")
    # BLEU skoru hesaplama
    bleu_result = bleu.compute(predictions=predictions, references=references)
    print("BLEU skoru hesaplandı.")
    # ROUGE skoru hesaplama
    rouge_result = rouge.compute(predictions=predictions, references=references)
    print("ROUGE skoru hesaplandı.")
    # METEOR skoru hesaplama
    meteor_result = meteor.compute(predictions=predictions, references=references)
    print("METEOR skoru hesaplandı.")
    # BERTScore skoru hesaplama
    bertscore_result = bertscore.compute(
        predictions=predictions, references=references, lang="tr"
    )
    print("BERTScore skoru hesaplandı.")
    # Tüm metrikleri birleştirme
    metrics = {
        "bleu": bleu_result,
        "rouge": rouge_result,
        "meteor": meteor_result,
        "bertscore": bertscore_result,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Metrikleri hesapla ve sonuçları kaydet"
    )
    parser.add_argument(
        "merged_file", type=str, help="Birleştirilmiş CSV dosyasının yolu"
    )
    parser.add_argument(
        "keys", type=str, nargs="+", help="Metriklerin hesaplanacağı anahtarlar"
    )

    args = parser.parse_args()

    # Dosyanın yüklenmesi
    df = pd.read_csv(args.merged_file)

    for key in args.keys:
        if key not in df.columns:
            raise ValueError(f"Anahtar '{key}' dosyada bulunamadı.")

        predictions = df[key].dropna().tolist()
        references = df["answer"].dropna().tolist()

        metrics = compute_metrics(predictions, references)

        # Sonuçları JSON dosyasına kaydetme
        output_file = f"{key}_result.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Sonuçlar kaydedildi: {output_file}")


if __name__ == "__main__":
    main()
