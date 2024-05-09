import csv
import json
from veri_setleri.degiskenler import diyabet_var, diyabet_yok, ANA_VERI_SETI
import pandas as pd
import os

def shuffle_and_save(filename):
    # Dosyayı DataFrame olarak yükle
    df = pd.read_csv(filename)
    
    # DataFrame'in satırlarını karıştır
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    shuffled_filename = f'shuffled_{filename}'
    # Karıştırılmış verileri dosyaya geri yaz
    shuffled_df.to_csv(shuffled_filename, index=False)

# JSON dosyasını oku
try:
    with open('ana_veri_seti_icerik.json', 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        min_count = min(data.values())
except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")
    exit()

# En düşük veri grubunun %80'ini ve %20'sini hesapla
train_limit = int(min_count * 0.8)
test_limit = min_count - train_limit


# Sayıcılar
train_var_count = 0
train_yok_count = 0
test_var_count = 0
test_yok_count = 0
# CSV dosyalarını oku ve yaz
try:
    with open(ANA_VERI_SETI, 'r', encoding='utf-8') as csvfile, \
         open('train.csv', 'w', newline='', encoding='utf-8') as trainfile, \
         open('test.csv', 'w', newline='', encoding='utf-8') as testfile:
        
        reader = csv.reader(csvfile)
        train_writer = csv.writer(trainfile, quoting=csv.QUOTE_ALL)
        train_writer.writerow(["text"])
        test_writer = csv.writer(testfile, quoting=csv.QUOTE_ALL)
        test_writer.writerow(["text"])
        for row in reader:
            row_text = ' '.join(row)
            if train_var_count < train_limit and any(phrase in row_text for phrase in diyabet_var):
                train_writer.writerow([row_text])
                train_var_count += 1
            elif train_yok_count < train_limit and any(phrase in row_text for phrase in diyabet_yok):
                train_writer.writerow([row_text])
                train_yok_count += 1
            elif test_var_count < test_limit and any(phrase in row_text for phrase in diyabet_var):
                test_writer.writerow([row_text])
                test_var_count += 1
            elif test_yok_count < test_limit and any(phrase in row_text for phrase in diyabet_yok):
                test_writer.writerow([row_text])
                test_yok_count += 1

            # Eğer her kategoriden yeterli veri toplandıysa döngüden çık
            if (train_var_count == train_limit and train_yok_count == train_limit and
                test_var_count == test_limit and test_yok_count == test_limit):
                break

except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")
    exit()



shuffle_and_save('test.csv')
shuffle_and_save('train.csv')