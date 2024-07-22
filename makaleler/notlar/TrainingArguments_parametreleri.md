# GPT-4 hiperparametre açıklamaları

* **per_device_train_batch_size:** Her cihazda eğitilecek örnek sayısıdır. Daha büyük batch size, daha hızlı eğitim sağlar ancak daha fazla bellek gerektirir.
* **gradient_accumulation_steps** Modelin parametre güncellemelerinden önce kaç adımda gradyanların biriktirileceğini belirler. Bu, efektif batch size'ı artırmak için kullanılır.
* **warmup_steps** Öğrenme oranının tam değerine ulaşmadan önceki adım sayısıdır. Bu, modelin daha stabil öğrenmesine yardımcı olur. Öğrenme oranı çizelgelerinde kullanılır ve model eğitiminin başında öğrenme oranının kademeli olarak artırılmasını sağlar.
* **num_train_epochs** Eğitim epoch sayısı,
* **learning_rate** Modelin her adımda ne kadar büyük bir adım atacağını belirler. Öğrenme oranı, eğitim hızını ve stabilitesini etkiler.
* **lr_scheduler_type** Öğrenme oranının zamanla nasıl değişeceğini belirler. "linear" gibi scheduler tipleri, öğrenme oranını belirli bir hızda düşürür.
* **logging_steps** Eğitim sırasında kaç adımda bir eğitim durumunun loglanacağını belirler. Eğitim sürecini izlemeye yardımcı olur.
* **weight_decay** Model ağırlıklarının azalmasını teşvik ederek aşırı uyumu (overfitting) önlemeye yardımcı olur.
* **seed** Rastgelelik içeren işlemler için başlangıç değeri sağlar. Bu, deneylerin tekrarlanabilir olmasını sağlar.
* **optim** Kullanılacak optimizasyon algoritmasını belirtir. Örneğin, "adamw_8bit" bellek kullanımını optimize eder.
* **fp16** Yarı hassas (16 bit) floating point hesaplamalarını etkinleştirir. Bellek kullanımını azaltır ve hesaplama hızını artırır.
* **bf16** Bfloat16 (16 bit) floating point hesaplamalarını etkinleştirir. FP16'ya benzer avantajlar sunar ancak daha stabil olabilir.

# Benim kullanacağım değerler ve sebepleri

* seed=3407 zaten pek mühim değil aynı işin tekrarlanabilmesi için
* learning_rate=1e-4
**KAYNAK** Towards Learning Universal Hyperparameter Optimizers with Transformers Figure 1
* warmup_steps=2000 bu tamamen deneysel seçilecek
* gradient_accumulation_steps cihaza göre seçilecek önemsiz
* per_device_train_batch_size cihaza göre seçilecek önemsiz
* logging_steps=2000 bilgi amaçlı olduğu için önemsiz
* save_strategy="steps" eval_strategy="steps" logging_strategy="steps"
* num_train_epochs=1 ya da 2 bu da deneysel
* max_seq_length=modelin boyutu kadar
* lr_scheduler_type="linear" olacak.
* gradient_accumulation_steps=32
* weight_decay=0.01
**KAYNAK** Exploring the Efficacy of Pre-trained Checkpoints in Text-to-Music Generation Task
Except for BART, the decoder for all other models is randomly initialised with the same configuration as the RND
encoder. Since almost every character in the ABC notation is
semantically independent, we took character-level tokenization (but added some common notations), with a vocabulary
size of 164. We trained all models using the same learning
rate α = 10−4
(for BART-large, it is 5×10−5
), with a 1,000-
step linear warmup and learning rate decay. We trained a total of 20 epochs with a batch size of 8, using the AdamW
optimizer with β1 = 0.9, β2 = 0.999,  = 10−8
, and a
weight decay coefficient of 0.01.

* optim="adamw_8bit" kullancam çünkü eğitim hızlanıyor, bellek kullanımı azalıyor ve performans umursanacak seviyede düşmüyor.
**KAYNAK** 8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION
We develop the first optimizers that use 8-bit statistics while maintaining the performance levels of using 32-bit optimizer states.
In Table 1, we see that 8-bit optimizers match replicated 32-bit performance for all tasks.
The broad range of tasks and competitive results demonstrate that 8-bit optimizers are a robust and effective replacement for 32-bit optimizers, do not require any additional changes in hyperparameters, and save a significant amount of memory while speeding up training slightly.
* bf16=True olacak ve fp16=False çünkü 
**KAYNAK** Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations
**GENİŞ DİNAMİK ARALIK** bfloat16 (BF16) is a new floating-point format [12] that is gaining traction due to its ability to work well in machine learning algorithms, in particular deep learning training. In contrast to the IEEE754-standardized 16bit (FP16) variant, BF16 does not compromise at all on range when being compared to FP32. As a reminder, FP32 numbers have 8 bits of exponent and 24 bits of mantissa (one implicit). BF16 cuts 16 bits from the 24-bit FP32 mantissa to create a 16-bit floating point datatype. In contrast FP16, roughly halves the FP32 mantissa to 10 explicit bits and has to reduce the exponent to 5 bits to fit the 16-bit datatype envelope

**DAHA AZ HASSASİYET KAYBI** Although BF16 offers less precision than FP16, it is better
suited to support deep learning tasks. As shown in [11], FP16’s range is not enough to accomplish deep learning training outof-the-box due to its limited range. BF16 does not suffer from this issue and the limited precision actually helps to generalize the learned weights in the neural net training task. In other words, lower precision can be seen as offering a builtin regularization property
**DERIN ÖĞRENME İÇİN UYGUNLUK** Additionally, the heart of deep learning is matrix multiplication. That means computing inner products of vectors of various length. Normally the dimensions of these vectors are pretty long: several hundreds to tens of thousands. Therefore, the community has settled on mixed-precision fused-multiply-add (FMA) hardware units.
**Karışık Hassasiyetli Hesaplama** One particular FMA operation that multiplies two BF16 numbers while accumulating in FP32 has been found useful in deep learning, where BF16 is the 16-bit floating point datatype with IEEE FP32 numerical range but 8 significant bits of precision.
**Yüksek Performanslı Hesaplamalar**We first demonstrate that computations of vector inner products and by natural extension, matrix-matrix products can be achieved by decomposing FP32 numbers in several BF16 numbers followed by appropriate computations that can accommodate the dynamic range and preserve accuracy compared to standard FP32 computations, while projecting up to 5.2x speed-up.
**Hata Analizi ve Performans İyileştirmeleri** The first observation one might make is that this last case, a triplet of BF16s, is comparable to FP32 as we have identical range and mantissa bits. Recently such an idea was also employed for NVIDIA Tensorcores with two FP16 numbers for FFT [6]. However more mechanics are needed due to lower ranges of FP16 and only 22 bits total mantissa (if counting the implicit bits.)"
## GPT'nin bf16 vs fp16 vs fp32 yazısı
### BF16'nın FP16'ya Göre Avantajları
* Geniş Dinamik Aralık: BF16, FP32'nin 8 bitlik üssü ile aynı üssü kullanır, bu da FP16'nın 5 bitlik üssüne göre önemli bir avantaj sağlar. Bu sayede BF16, FP32 ile aynı dinamik aralığı sunar ve bu da derin öğrenme eğitiminde doğrudan kullanımı mümkün kılar​(Leveraging_the_bfloat16…)​.

* Daha Az Hassasiyet Kaybı: FP16, 10 bitlik mantisaya sahiptir ve bu da hassasiyet kaybına yol açar. BF16 ise, FP32'nin mantisasının sadece 16 bitini kullanır, bu da FP16'ya göre daha az hassasiyet kaybı demektir​(Leveraging_the_bfloat16…)​.

* Derin Öğrenme için Uygunluk: BF16'nın daha düşük hassasiyeti, derin öğrenme görevlerinde ağırlıkların genelleştirilmesine yardımcı olur, bu da modelin aşırı öğrenmesini (overfitting) engelleyebilir​(Leveraging_the_bfloat16…)​.

### BF16'nın FP32 ile Benzer Performansı
* Karışık Hassasiyetli Hesaplama: BF16, FP32 ile karışık hassasiyetli hesaplamalarda kullanılabilir. Örneğin, iki BF16 sayısını çarparken sonuç FP32 formatında toplanabilir. Bu, 16 bitlik çarpım sonuçlarının tam olarak korunmasını ve 24 bit hassasiyetle toplanmasını sağlar​(Leveraging_the_bfloat16…)​.

* Yüksek Performanslı Hesaplamalar: BF16, daha küçük mantisa boyutu nedeniyle daha yüksek hesaplama performansı sunar. Örneğin, FP32'ye göre 5.2 kat daha hızlı matris çarpımları (GEMM) gerçekleştirebilir​(Leveraging_the_bfloat16…)​.

* Hata Analizi ve Performans İyileştirmeleri: BF16 ile yapılan hesaplamalar, belirli durumlarda FP32'den daha hassas sonuçlar verebilir. Üç BF16 sayısının birleştirilmesiyle yapılan hesaplamalar, FP32 hesaplamalarına kıyasla benzer veya daha iyi sonuçlar elde edebilir​(Leveraging_the_bfloat16…)​.

