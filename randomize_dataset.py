import argparse
import pandas as pd
import os

def shuffle_csv(input_filepath, output_filepath):
    # CSV dosyasını oku
    df = pd.read_csv(input_filepath)
    
    # Verileri karıştır
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Karıştırılmış verileri yeni dosyaya yaz
    df_shuffled.to_csv(output_filepath, index=False)
    print(f"Karıştırılmış dosya '{output_filepath}' olarak kaydedildi.")

if __name__ == "__main__":
    # Argümanları ayarla
    parser = argparse.ArgumentParser(description="Bir CSV dosyasını karıştır ve 'random_' ön ekiyle yeni bir dosyaya kaydet.")
    parser.add_argument('input_filepath', type=str, help='Giriş CSV dosyasının yolu')
    
    # Argümanları al
    args = parser.parse_args()
    
    # Giriş dosyasının yolunu al
    input_filepath = args.input_filepath
    
    # Çıkış dosyasının yolunu belirle
    output_filepath = os.path.join(os.path.dirname(input_filepath), 'random_' + os.path.basename(input_filepath))
    
    # CSV dosyasını karıştır ve yeni dosyaya yaz
    shuffle_csv(input_filepath, output_filepath)
