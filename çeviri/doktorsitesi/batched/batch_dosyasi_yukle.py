import sys
sys.path.append("../..")
from fonksiyonlar import veri_yolu_al

from openai import OpenAI

with open("../../../api_key.txt", "r") as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)
veri_yolu = veri_yolu_al()

batch_input_file = client.files.create(
  file=open(veri_yolu, "rb"),
  purpose="batch"
)
with open("batch_file_info.txt", "w") as file:
    file.write(str(batch_input_file))