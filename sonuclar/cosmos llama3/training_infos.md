# Promt Format
```
alpaca_prompt = """Sen bir doktorsun. Soruları buna göre cevapla.
### <|reserved_special_token_0|>:
{}

### <|reserved_special_token_1|>:
{}"""
```

# Training args
```
batch_size = 128
gradient_accumulation_steps = 32
num_train_epochs = 2
per_device_batch_size = int(batch_size / gradient_accumulation_steps)
training_args = TrainingArguments(
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        save_total_limit = 1,
        warmup_steps = int(2000 / batch_size),
        num_train_epochs = num_train_epochs,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        save_strategy = "steps",
        eval_strategy = "steps",
        logging_strategy = "steps",
        save_steps = int(5000 / batch_size * num_train_epochs),
        eval_steps = int(28900 / batch_size * num_train_epochs),
        logging_steps = int(28900 / batch_size * num_train_epochs),
    )
```

# Trainer args

```
max_seq_length = 8192
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)
```

# From pretrained args

```
from unsloth import FastLanguageModel
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = output_dir,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```