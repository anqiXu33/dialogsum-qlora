# Experiment runbook (Kaggle)

How to run every experiment in the improvement plan. Each block is one Kaggle
notebook cell. Use **GPU T4 x2** or **P100** and turn **Internet on**.
Kaggle sessions stop after 12 h, so the long runs are split into separate sessions.

## Setup (first cell of every session)

```python
!git clone https://github.com/anqiXu33/dialogsum-qlora.git
%cd dialogsum-qlora
!pip install -q -U transformers peft trl bitsandbytes datasets accelerate evaluate rouge_score bert_score
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
login()   # needed for push_to_hub.py and for --revision on private repos
```

To keep `runs/` and `results/` between sessions, save the notebook version with
"Save & Run All" or download `results/` at the end of each session.

## Stage 0: freeze v1

```python
!python push_to_hub.py --tag-current v1-3k-1epoch
```

## Stage 1: v2 and fair baselines (session 1, about 9 h)

```python
# Baselines on the full test set (1,500 dialogues)
!python run_eval.py --name base-0shot
!python run_eval.py --name base-1shot --shots 1
!python run_eval.py --name v1 --adapter AnqiXaq/dialogsum-qlora-adapter --revision v1-3k-1epoch
```

Check `results/base-0shot.json` → `format_followed`. If it is far below 100 %, the
zero-shot base was partly penalised for format, and `base-1shot` is the fairer baseline.

```python
# v2: all 12,460 training dialogues, loss on the summary only
!python train.py --out-dir runs/v2
!python run_eval.py --name v2 --adapter runs/v2/adapter
!python compare.py
```

If v2 beats v1, publish it:

```python
!python push_to_hub.py --adapter runs/v2/adapter --tag v2-full
```

## Stage 2: ablations (session 2, about 8 h)

Same seed and the same shuffled subset for all runs, so only one factor changes.

```python
# Loss masking: identical data, full-sequence loss vs summary-only loss
!python train.py --max-train 3000 --loss full       --out-dir runs/n3000-full
!python train.py --max-train 3000 --loss completion --out-dir runs/n3000
!python run_eval.py --name n3000-full --adapter runs/n3000-full/adapter
!python run_eval.py --name n3000      --adapter runs/n3000/adapter
```

```python
# Data scaling (12,460 = v2 from stage 1)
for n in [500, 1000]:
    !python train.py --max-train {n} --out-dir runs/n{n}
    !python run_eval.py --name n{n} --adapter runs/n{n}/adapter
!python compare.py
```

Optional, if quota is left: a 7B zero-shot reference point.

```python
!python run_eval.py --name qwen7b-0shot --model Qwen/Qwen2.5-7B-Instruct --batch-size 8
```

## What to send back

The `results/` folder (all `.json` and `.jsonl` files) and `runs/*/run_info.json`
(stored in the repo as `results/training_logs/<run>.json`). Download a whole run with
`kaggle kernels output <user>/<notebook> -p kaggle/<stage>`; `kaggle/` is git-ignored.
These are enough to build the tables, the data-scaling plot and the error analysis.
