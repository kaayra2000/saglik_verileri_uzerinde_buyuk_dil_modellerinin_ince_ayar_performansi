import os
import cairosvg

def svg_to_png(folder_path):
    # Klasördeki tüm dosyaları al
    for filename in os.listdir(folder_path):
        # SVG dosyalarını kontrol et
        if filename.endswith('.svg'):
            svg_file = os.path.join(folder_path, filename)
            png_file = os.path.join(folder_path, filename[:-4] + '.png')
            # Dönüştürme işlemini yap
            cairosvg.svg2png(url=svg_file, write_to=png_file)
            print(f'{filename} dosyası {png_file} olarak kaydedildi.')

# Kullanım
folder_path = '.'  # Buraya dönüştürmek istediğiniz klasörün yolunu yazın
svg_to_png(folder_path)
