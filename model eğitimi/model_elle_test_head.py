from  transformers  import  AutoTokenizer, AutoModelWithLMHead

model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
context = """
'War and Peace' is a classic novel written by the Russian author Leo Tolstoy. 
It is considered one of the greatest works of world literature and was first published in 1869. 
The novel tells the story of five aristocratic families during the Napoleonic era, 
and it explores themes of love, war, and the human condition.
"""
def cevap_uret(metin):
    input = f"question: {metin} context: {context}"
    encoded_input = tokenizer([input],
                                return_tensors='pt',
                                max_length=512,
                                truncation=True)
    output = model.generate(input_ids = encoded_input.input_ids,
                                attention_mask = encoded_input.attention_mask)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)

while True:
    # Girdi al
    metin = input("Lütfen bir metin girin (çıkmak için 'exit' yazın): ")
    # Çıkış koşulu
    if metin.lower() == 'exit':
        break
    # Girdiyi modelden geçir
    cevap_uret(metin)