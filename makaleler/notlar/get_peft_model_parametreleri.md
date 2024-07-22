# GPT-4 hiperparametre açıklamaları

* **r (rank):** Bu, düşük dereceli matrislerin boyutunu belirler. Küçük r değerleri modelin ince ayar sırasında daha az parametre kullanmasını sağlar, ancak bu durumda hassasiyet azalabilir. Daha karmaşık görevler için r değerini artırmanız gerekebilir​
* **lora_alpha:** Bu, düşük dereceli adaptasyon matrislerinin öğrenme hızını belirler. Genellikle 16 iyi bir başlangıç değeridir. Bu değeri, modelin performansını ve hesaplama kaynaklarınızı göz önünde bulundurarak ayarlayabilirsiniz​. lora_alpha **büyük** olursa overfitting olur. Rank ve lora_alpha ayarlamak **α/r** öğrenme oranını ayarlamaya benzer.
* **lora_dropout:** Bu parametre, modelin aşırı uyumunu önlemek için rastgele nöronların devre dışı bırakılma oranını belirler. 0 değeri, dropout kullanmayacağınız anlamına gelir ve genellikle optimize edilmiş bir ayardır
* **bias:** Bias parametresinin "none" olarak ayarlanması, eklenen adaptasyon katmanlarında bias kullanılmayacağı anlamına gelir ve bu, genellikle en iyi performansı sağlar​
* **use_gradient_checkpointing:** "unsloth" seçeneği, bellek kullanımını optimize ederek daha büyük batch boyutlarıyla çalışmanıza olanak tanır. Bu, çok uzun dizilerle çalışırken yararlıdır​ 
* **random_state:** Bu, rastgele işlemler için bir başlangıç değeri belirler ve sonuçların tekrarlanabilir olmasını sağlar. Değerin 3407 olarak ayarlanması genellikle yeterlidir.
* **use_rslora:** Rank Stabilized LoRA (rslora) kullanılıp kullanılmayacağını belirler. False olarak bırakabilirsiniz, çünkü bu genellikle ek bir optimizasyon gerektirir ve her zaman gerekli değildir​
* **loftq_config:** LoftQ, modelin kuantizasyonunu ayarlamak için kullanılır. Eğer bu özellik gerekli değilse, None olarak bırakabilirsiniz​

# Benim kullanacağım değerler ve sebepleri

* r=8 kullanacağım. Çünkü LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS adlı makalede tablo 6 ve  15'te bariz şekilde daha yüksek değerlere gerek olmadığı gözüküyor.

* lora_alpha=16 çünkü **lora_alpha/r** (bir üstteki r) genelde 2 ya da 4 olarak kullanılıyor. Bu değer arttıkça overfitting olur.
**KAYNAK:** LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
As shown in Section 7.3, for r = 4, this amplification factor is as large as 20. In other words, there are (generally speaking) four feature directions in each layer (out of the entire feature space from the pre-trained model W), that need to be amplified by a very large factor 20, in order to achieve our reported accuracy for the downstream specific task.

* lora_dropout=0 çünkü overfitting olmayacağını düşünüyorum
**KAYNAK:** LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
We use LoRA (Hu et al., 2021) with rank 16 for all tasks with a dropout rate of 0.
We set the dropout rate to 0 unless otherwise specified.
* bias=none çünkü Bias'ları eğitmek yerine onları kaldırmak, modelin eğitiminde daha az parametre anlamına gelir, bu da daha hızlı eğitim ve daha az hesaplama gereksinimi demektir.
* use_gradient_checkpointing="unsloth" zaten pek bir olayı yok
* random_state=3407 önemsiz
* use_rslora=False zaten loranın özelleşmiş hali kullanmayacağım
* loftq_config=None LoftQ (LoRA-Fine-Tuning-aware Quantization) zaten bu da aynı