import pandas as pd
import argparse


def merge_responses(test_file, doktor_file, key_column):
    # Dosyaların yüklenmesi
    test_df = pd.read_csv(test_file)
    doktor_df = pd.read_csv(doktor_file)

    # İndise göre doktor_df'in response kolonunu test_df'e ekle
    test_df[key_column] = None
    for _, row in doktor_df.iterrows():
        idx = row["index"]
        if idx in test_df.index:
            test_df.at[idx, key_column] = row["response"]

    # Dosyayı kaydet
    output_file = "merged_" + test_file
    test_df.to_csv(output_file, index=False)
    print(f"Yeni dosya oluşturuldu ve kaydedildi: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Dosyaları birleştir ve yeni bir kolon ekle"
    )
    parser.add_argument("test_file", type=str, help="Test CSV dosyasının yolu")
    parser.add_argument("doktor_file", type=str, help="Doktor CSV dosyasının yolu")
    parser.add_argument("key_column", type=str, help="Eklenecek kolonun anahtarı")

    args = parser.parse_args()

    merge_responses(args.test_file, args.doktor_file, args.key_column)


if __name__ == "__main__":
    main()
