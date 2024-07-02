# Training Compute-Optimal Large Language Models

We use a maximum learning rate of 2 × 10 −4 for the smallest models and 1 . 25 × 10 −4 for the largest
models. In all cases, the learning rate drops by a factor of 10× during training, using a cosine schedule.
We make the assumption that the cosine cycle length should be approximately matched to the number
of training steps. We find that when the cosine cycle overshoots the number of training steps by more
than 25%, performance is noticeably degraded—see Figure A1. 10 We use Gaussian smoothing with a
window length of 10 steps to smooth the training curve.



# MFA Introducing cosmosGPT: Monolingual Training for Turkish Language Models

* 1e−4 linear kullanmış

The training of the models was conducted using Google
Cloud’s TPUv3-8 infrastructure. The Large model was trained
with a batch size of 16 and for 2 epochs, while the Medium
model used a batch size of 128 and was trained for 3 epochs.
The training process was optimized using an Adam optimizer
with a learning rate of 1e−4 with linear decay. This approach
has enabled both rapid and efficient training of the models.
In conclusion, this training process and the use of highquality datasets have enabled us to achieve significant advancements in the accuracy and reliability of our Turkish language
models.

# Automated question generation and question answering from Turkish texts

* 1e−4 ve 1e-3 linear kullanmış

3.2. Hyper-parameter Tuning

We experimentally evaluated mT5 [29] against BERTurk [21] and, to have a fair comparison, we performed hyper
parameter tuning. For both models, we used grid-search to select the best optimizer type (Adafactor, AdamW),
initial learning rate (1e-3, 1e-4, 1e-5 ) and number of training epochs (1, 3, 5, 8, 10, 15, 20). BERTurk-base
language model [21] has been fine-tuned for QA task on TQuADv2 training split, F1 and EM scores have been
calculated on TQuADv2 validation split and XQuAD Turkish split. We selected the set of parameters which
attain the overall best scores in all metrics: AdamW optimizer with a learning rate of 1e-4 and number of