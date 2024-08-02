import pandas as pd
import argparse


def create_filtered_csv(merged_file, insana_sorulacak_file, output_file):
    # Dosyaların yüklenmesi
    merged_df = pd.read_csv(merged_file)
    insana_sorulacak_df = pd.read_csv(insana_sorulacak_file)

    # İndisleri kullanarak filtreleme
    filtered_df = merged_df.loc[insana_sorulacak_df.index]

    # Yeni dosyayı kaydet
    filtered_df.to_csv(output_file, index=False)
    print(f"Yeni dosya oluşturuldu ve kaydedildi: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="İndislerle filtreleme ve yeni CSV oluşturma"
    )
    parser.add_argument(
        "merged_file", type=str, help="Birleştirilmiş CSV dosyasının yolu"
    )
    parser.add_argument(
        "insana_sorulacak_file",
        type=str,
        help="İndis bilgilerini içeren CSV dosyasının yolu",
    )
    parser.add_argument(
        "output_file", type=str, help="Oluşturulacak yeni CSV dosyasının yolu"
    )

    args = parser.parse_args()

    create_filtered_csv(args.merged_file, args.insana_sorulacak_file, args.output_file)


if __name__ == "__main__":
    main()
