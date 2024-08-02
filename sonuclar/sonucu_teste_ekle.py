import pandas as pd
import argparse


def merge_responses(test_file, doktor_files, key_columns):
    # Dosyaların yüklenmesi
    test_df = pd.read_csv(test_file)

    for doktor_file, key_column in zip(doktor_files, key_columns):
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
    parser.add_argument(
        "doktor_files_and_keys",
        type=str,
        nargs="+",
        help="Diğer dosyalar ve onların anahtarları (her dosya için bir anahtar)",
    )

    args = parser.parse_args()

    if len(args.doktor_files_and_keys) % 2 != 0:
        raise ValueError("Her dosya için bir anahtar belirtilmelidir.")

    doktor_files = args.doktor_files_and_keys[::2]
    key_columns = args.doktor_files_and_keys[1::2]

    merge_responses(args.test_file, doktor_files, key_columns)


if __name__ == "__main__":
    main()
