# DialogSum qLoRA — interaction summariser

A small **Qwen2.5** model fine-tuned with **qLoRA** (4-bit NF4 base + LoRA adapter)
to condense a multi-turn conversation into a **structured summary** (topic + summary).

The model reads a long exchange and emits a short, structured record — a compact
take on interaction condensation. Trained on a single Colab T4, evaluated with
ROUGE + BERTScore, and shipped as an interactive Hugging Face Space.

## Repo structure

```
dialogsum-qlora/
├── README.md           this file
├── requirements.txt    training deps (Colab)
├── data.py             prompt formatting (shared)
├── train.py            qLoRA fine-tuning  -> saves the adapter
├── inference.py        load base + adapter, summarise one conversation
├── evaluate.py         ROUGE + BERTScore on the test split (base vs tuned)
└── app.py              Hugging Face Space demo (ZeroGPU)
```

## How to run

1. **Train** — on Colab (Runtime → T4 GPU):
   ```bash
   pip install -r requirements.txt
   python train.py
   ```
   Finishes in roughly tens of minutes on 3000 examples. Produces
   `qlora-dialogsum-adapter/` (a few MB).

2. **Sanity check** — `python inference.py` should print a `Topic: ... / Summary: ...`.

3. **Evaluate** — `python evaluate.py` for the tuned model, then set
   `USE_ADAPTER=False` and run again for the base model, to measure the
   before/after delta.

4. **Demo (optional)** — push the adapter to the Hub, create a Gradio Space with
   **ZeroGPU** hardware, drop in `app.py` (set `ADAPTER` to your Hub repo id), and
   add a Space `requirements.txt` (see below).

## Model options

| Model | When to use |
|-------|-------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | fastest; pick if you just want the loop to run quickly |
| `Qwen/Qwen2.5-1.5B-Instruct` | **default** — good quality/speed balance on a T4 |
| `Qwen/Qwen2.5-3B-Instruct`   | higher quality; still fits in 4-bit on a T4 |

## Demo (Hugging Face Space)

The Space needs its own `requirements.txt` (no bitsandbytes — fp16 on ZeroGPU):

```
torch
transformers
peft
gradio
spaces
```

And the Space `README.md` needs YAML front-matter at the very top:

```
---
title: DialogSum qLoRA Summariser
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
hardware: zero-gpu
---
```

## Design notes

- **Structured output** — the training target is built as `topic` + `summary`,
  reusing DialogSum's existing `topic` field, so the model emits a small structured
  record with no extra labelling.
- **Evaluation** — ROUGE for surface n-gram overlap, BERTScore for semantic
  similarity; comparing base vs. fine-tuned measures what the adapter actually added.
- **T4 note** — the 4-bit compute dtype is `float16` (Turing GPUs have no bf16);
  switch to `bfloat16` only on Ampere or newer GPUs.
