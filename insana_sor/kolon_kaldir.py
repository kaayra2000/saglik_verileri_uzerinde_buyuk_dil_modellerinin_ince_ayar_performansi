import pandas as pd
import argparse


def remove_column_and_save(csv_file, column_key):
    # Dosyanın yüklenmesi
    df = pd.read_csv(csv_file)

    # Kolonu silme
    if column_key in df.columns:
        df.drop(columns=[column_key], inplace=True)
    else:
        raise ValueError(f"Kolon '{column_key}' dosyada bulunamadı.")

    # Dosyayı aynı isimle kaydetme
    df.to_csv(csv_file, index=False)
    print(f"'{column_key}' kolonu silindi ve dosya kaydedildi: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Kolon silme ve dosyayı kaydetme")
    parser.add_argument("csv_file", type=str, help="CSV dosyasının yolu")
    parser.add_argument("column_key", type=str, help="Silinecek kolonun anahtarı")

    args = parser.parse_args()

    remove_column_and_save(args.csv_file, args.column_key)


if __name__ == "__main__":
    main()
