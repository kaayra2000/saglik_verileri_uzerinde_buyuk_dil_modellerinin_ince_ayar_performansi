import pandas as pd
import argparse


# CSV dosyasını parça parça okuyacak bir fonksiyon tanımlayın
def process_csv_in_chunks(input_filepath, output_filepath, chunk_size=1000):
    # Çıktı dosyasını aç
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        outfile.write("text\n")
        # CSV dosyasını parça parça okuyun
        for chunk in pd.read_csv(input_filepath, chunksize=chunk_size):
            # Her parça için chat şablonu uygulayın
            chunk["text"] = chunk.apply(
                lambda row: apply_chat_template(
                    row["question_content"],
                    row["question_answer"],
                    row["doctor_title"],
                    row["doctor_speciality"],
                ),
                axis=1,
            )
            # Sonuçları yeni dosyaya yazın
            chunk["text"].to_csv(outfile, index=False, header=False, mode="a")


def apply_chat_template(
    question_content,
    question_answer,
    doctor_title="bilinmiyor",
    doctor_speciality="bilinmiyor",
):
    user_message = f"<|USER|> {question_content}\n"
    doctor_info = (
        f"<|DOCTOR_TITLE|> {doctor_title} <|SPECIALITY|> {doctor_speciality}\n"
    )
    assistant_message = f"<|ASSISTANT|> {question_answer} </s>\n"
    return f"{user_message}\n{doctor_info}\n{assistant_message}"


if __name__ == "__main__":
    # Argümanları ayarlayın
    parser = argparse.ArgumentParser(description="CSV dosyasını parça parça işleyin.")
    parser.add_argument(
        "input_filepath",
        type=str,
        help="Giriş CSV dosyasının yolu",
        default="sohbetler20000.csv",
    )

    # Argümanları alın
    args = parser.parse_args()

    # Giriş ve çıkış dosyalarının yollarını alın
    input_filepath = args.input_filepath
    output_filepath = "templated_" + input_filepath

    # CSV dosyasını parça parça işleyin
    process_csv_in_chunks(input_filepath, output_filepath)
