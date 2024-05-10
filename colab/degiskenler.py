import os
base_path = "/content/drive/My Drive/yl_tez"
lr = 0.001
bs = 32

epoch_sayisi = 20
max_len = 128
# anahatlar
metin = "text"
label = "diabetes"
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
veri_seti_adi = "ana_veri_seti.csv"
#veri_seti_adi =  "ana_veri_seti_deneme.csv"
# Veri seti dosya yolu
data_filepath = os.path.join(base_path, veri_seti_adi)
# Dosya adları
model_adi = "BestModelParameters"
checkpoint_path_name = "checkpoint"
checkpoint_name = 'checkpoint.pth'
checkpoint_pth = os.path.join(base_path, checkpoint_path_name)
checkpoint_pth_peft = checkpoint_pth + "_peft"
model_yolu = os.path.join(base_path, model_adi)
model_yolu_peft = model_yolu + "_peft"
gorsel_klasor_adi = "gorseller"
gorsel_yolu = os.path.join(base_path, gorsel_klasor_adi)
SORU_TOKEN ="<|soru|>"
CEVAP_TOKEN ="<|cevap|>"