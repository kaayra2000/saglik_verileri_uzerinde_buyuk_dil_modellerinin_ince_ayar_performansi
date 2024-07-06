import sys

from openai import OpenAI

with open("../../../api_key.txt", "r") as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

batch_input_file_id = "file-LGnKPscHXqLSJ8mFH0YfkBMW"

sonuc = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "20k doktorsitesi"
    }
)

with open("batch_info.txt", "w") as file:
    file.write(str(sonuc))