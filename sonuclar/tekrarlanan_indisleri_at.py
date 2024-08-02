import pandas as pd
import argparse


def remove_duplicate_indices(chat_doctor_file):
    # Dosyanın yüklenmesi
    df = pd.read_csv(chat_doctor_file)

    # Tekrarlanan indeksleri kaldır
    df = df[~df.duplicated(subset=["index"], keep="first")]

    # Dosyayı aynı isimle kaydet
    df.to_csv(chat_doctor_file, index=False)
    print(f"Tekrarlanan indeksler silinmiş ve dosya kaydedilmiştir: {chat_doctor_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Tekrarlanan indeksleri sil ve dosyayı kaydet"
    )
    parser.add_argument(
        "chat_doctor_file", type=str, help="Chat-Doktor CSV dosyasının yolu"
    )

    args = parser.parse_args()

    remove_duplicate_indices(args.chat_doctor_file)


if __name__ == "__main__":
    main()
