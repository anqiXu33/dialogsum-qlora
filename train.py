"""qLoRA fine-tune of Qwen2.5-1.5B-Instruct on DialogSum.

Defaults reproduce the v2 setup: full train split, loss on the summary only,
eval loss on a validation subset, best checkpoint kept. Tested on Kaggle GPUs.

Examples
--------
    python train.py                                   # v2: all data, completion-only loss
    python train.py --max-train 3000 --loss full \
        --out-dir runs/v1-repro                       # reproduce v1
    python train.py --max-train 1000 --out-dir runs/n1000   # data-scaling point
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from data import to_prompt_completion, to_training_text

DATASET_ID = "knkarthick/dialogsum"  # id, dialogue, summary, topic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out-dir", default="runs/v2")
    p.add_argument("--max-train", type=int, default=None, help="first N train examples (default: all 12,460)")
    p.add_argument("--max-eval", type=int, default=200, help="validation examples used for eval loss")
    p.add_argument("--loss", choices=["completion", "full"], default="completion",
                   help="completion: loss on Topic/Summary only. full: loss on every token (v1).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None, help="default: 2 x lora-r")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto",
                   help="auto: bf16 on Ampere+, fp16 on older GPUs such as T4/P100")
    return p.parse_args()


def main():
    args = parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    os.makedirs(args.out_dir, exist_ok=True)

    # Real bf16 needs compute capability >= 8 (Ampere+). T4 (7.5) reports bf16 as
    # "supported" but only emulates it, which is much slower, so use fp16 there.
    if args.precision == "auto":
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    else:
        use_bf16 = args.precision == "bf16"
    print(f"GPU: {torch.cuda.get_device_name(0)} | precision: {'bf16' if use_bf16 else 'fp16'}")
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train = load_dataset(DATASET_ID, split="train").shuffle(seed=args.seed)
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    val = load_dataset(DATASET_ID, split="validation").select(range(args.max_eval))

    if args.loss == "completion":
        fmt = to_prompt_completion
    else:
        fmt = lambda ex: to_training_text(ex, tokenizer)  # noqa: E731
    train = train.map(fmt, remove_columns=train.column_names)
    val = val.map(fmt, remove_columns=val.column_names)

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_length=args.max_length,
        completion_only_loss=(args.loss == "completion"),
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    if not use_bf16:
        # fp16 mixed precision needs fp32 master weights: the GradScaler cannot unscale
        # fp16/bf16 gradients. Keep only the (small) trainable LoRA weights in fp32;
        # the 4-bit base still computes in fp16. This is the standard QLoRA setup.
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    trainer.train()
    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.save_model(adapter_dir)

    run_info = {
        "args": vars(args),
        "n_train": len(train),
        "best_eval_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. Best eval loss: {trainer.state.best_metric}")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Next: python run_eval.py --adapter {adapter_dir} --name {os.path.basename(args.out_dir)}")


if __name__ == "__main__":
    main()
