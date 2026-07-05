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
├── run_eval.py         ROUGE + BERTScore on the test split (base vs tuned)
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

3. **Evaluate** — `python run_eval.py` for the tuned model, then set
   `USE_ADAPTER=False` and run again for the base model, to measure the
   before/after delta. (The script is named `run_eval.py`, not `evaluate.py`,
   so it doesn't shadow the pip `evaluate` library on `import evaluate`.)

4. **Demo (optional)** — push the adapter to the Hub, create a Gradio Space with
   **ZeroGPU** hardware, drop in `app.py` (set `ADAPTER` to your Hub repo id), and
   add a Space `requirements.txt` (see below).

## Results

Qwen2.5-1.5B, qLoRA adapter (1 epoch, 3000 examples), scored on 100 examples of
the DialogSum **test** split. Base = the untuned model on the same prompt.

| Metric        | Base   | qLoRA tuned | Δ      |
|---------------|:------:|:-----------:|:------:|
| ROUGE-1       | 0.205  | **0.367**   | +80%   |
| ROUGE-2       | 0.051  | **0.119**   | +134%  |
| ROUGE-L       | 0.159  | **0.293**   | +85%   |
| BERTScore F1  | 0.877  | **0.908**   | +3.2%  |

The adapter roughly **doubles ROUGE** — it teaches the model DialogSum's summary
style and the `Topic: … / Summary: …` format. BERTScore (semantics) starts high
and still gains ~3 points. Trained adapter:
[`AnqiXaq/dialogsum-qlora-adapter`](https://huggingface.co/AnqiXaq/dialogsum-qlora-adapter).

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
- **Precision** — chosen at runtime via `torch.cuda.is_bf16_supported()`: `bfloat16`
  on Ampere+ (and Kaggle), `float16` on Turing T4. No manual editing per GPU.
