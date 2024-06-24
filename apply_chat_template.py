import pandas as pd

# CSV dosyasını parça parça okuyacak bir fonksiyon tanımlayın
def process_csv_in_chunks(input_filepath, output_filepath, chunk_size=1000):
    # Çıktı dosyasını aç
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        outfile.write('text\n')
        # CSV dosyasını parça parça okuyun
        for chunk in pd.read_csv(input_filepath, chunksize=chunk_size):
            # Her parça için chat şablonu uygulayın
            chunk['text'] = chunk.apply(apply_chat_template, axis=1)
            # Sonuçları yeni dosyaya yazın
            chunk['text'].to_csv(outfile, index=False, header=False, mode='a')

def apply_chat_template(row):
    doctor_info = f"<DOCTOR_TITLE> {row['doctor_title']} <SPECIALITY> {row['doctor_speciality']}\n"
    user_message = f"<USER> {row['question_content']}\n"
    assistant_message = f"<ASSISTANT> {row['question_answer']} </s>\n"
    return f"{doctor_info}\n{user_message}\n{assistant_message}"

# Giriş ve çıkış dosyalarının yollarını belirleyin
input_filepath = 'sohbetler20000.csv'
output_filepath = "templated_" + input_filepath

# CSV dosyasını parça parça işleyin
process_csv_in_chunks(input_filepath, output_filepath)
