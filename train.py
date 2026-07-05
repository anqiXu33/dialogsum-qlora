"""qLoRA fine-tune of Qwen2.5-1.5B on DialogSum. Runs on a free Colab T4."""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from data import to_training_text

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # 0.5B / 3B also work, see README
DATASET_ID = "knkarthick/dialogsum"        # id, dialogue, summary, topic
MAX_TRAIN = 3000
OUT_DIR = "qlora-dialogsum-adapter"

# bf16 where the GPU supports it (Ampere+, and e.g. Kaggle), else fp16 (T4).
USE_BF16 = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

ds = load_dataset(DATASET_ID, split="train")
ds = ds.select(range(min(MAX_TRAIN, len(ds))))
ds = ds.map(
    lambda ex: to_training_text(ex, tokenizer),
    remove_columns=ds.column_names,
)

sft_config = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # effective batch 8
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=USE_BF16,                      # match the 4-bit compute dtype above
    fp16=not USE_BF16,
    logging_steps=20,
    save_strategy="epoch",
    max_length=1024,                    # newer trl renamed this from max_seq_length
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds,
    peft_config=lora_config,
    processing_class=tokenizer,         # older trl called this `tokenizer`
)

trainer.train()
trainer.save_model(OUT_DIR)
print(f"\nDone. Adapter saved to: {OUT_DIR}")
