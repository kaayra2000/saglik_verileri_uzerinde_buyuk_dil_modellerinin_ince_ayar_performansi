# Is Bigger Edit Batch Size Always Better? - An Empirical Study onModel Editing with Llama-3.

Makale 1024'ü önermiş ama ölçeklenebilirlik için vs. de katmış
sayfa 5'e bakılınca 64 optimum



# MFA Introducing cosmosGPT: Monolingual Training for Turkish Language Models

* büyük model için 16 batch size 2 epoch
* orta için 128 batch size 3 epoch

The training of the models was conducted using Google
Cloud’s TPUv3-8 infrastructure. The Large model was trained
with a batch size of 16 and for 2 epochs, while the Medium
model used a batch size of 128 and was trained for 3 epochs.
The training process was optimized using an Adam optimizer
with a learning rate of 1e−4 with linear decay. This approach
has enabled both rapid and efficient training of the models.
In conclusion, this training process and the use of highquality datasets have enabled us to achieve significant advancements in the accuracy and reliability of our Turkish language
models.