import pandas as pd
import argparse


def adjust_indices_and_save(doktor_mistral_file, first_index, last_index):
    # Dosyanın yüklenmesi
    df = pd.read_csv(doktor_mistral_file)

    # Belirtilen aralıktaki indeksleri 1 azalt
    df.loc[first_index:last_index, "index"] = (
        df.loc[first_index:last_index, "index"] - 1
    )

    # Aynı isimde dosyayı kaydet
    df.to_csv(doktor_mistral_file, index=False)
    print(f"İndisleri güncellenmiş dosya kaydedildi: {doktor_mistral_file}")


def main():
    parser = argparse.ArgumentParser(description="İndisleri ayarlama ve kaydetme")
    parser.add_argument(
        "doktor_mistral_file", type=str, help="İşlenecek CSV dosyasının yolu"
    )
    parser.add_argument("first_index", type=int, help="İlk indeks")
    parser.add_argument("last_index", type=int, help="Son indeks")

    args = parser.parse_args()

    adjust_indices_and_save(args.doktor_mistral_file, args.first_index, args.last_index)


if __name__ == "__main__":
    main()
