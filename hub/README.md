---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
datasets:
- knkarthick/dialogsum
metrics:
- rouge
- bertscore
tags:
- lora
- qlora
- sft
- trl
- summarization
- dialogue-summarization
model-index:
- name: dialogsum-qlora-adapter
  results:
  - task:
      type: summarization
      name: Dialogue Summarization
    dataset:
      name: DialogSum (test, 500 dialogues x 3 references)
      type: knkarthick/dialogsum
      split: test
    metrics:
    - type: rouge
      name: ROUGE-1
      value: 0.414
    - type: rouge
      name: ROUGE-2
      value: 0.157
    - type: rouge
      name: ROUGE-L
      value: 0.335
    - type: bertscore
      name: BERTScore F1
      value: 0.917
---

# DialogSum QLoRA adapter for Qwen2.5-1.5B-Instruct

A LoRA adapter that turns [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
into a dialogue summariser. Given a multi-turn conversation, it returns a short
**topic label** and a **one or two sentence summary**:

```
Topic: ask for help
Summary: Tom calls Sara because his daughter has a high fever. Sara agrees to look after his son.
```

Trained with QLoRA on all 12,460 training dialogues of
[DialogSum](https://github.com/cylnlp/dialogsum), one epoch on a single T4.

- **Code, ablations and error analysis:** [github.com/anqiXu33/dialogsum-qlora](https://github.com/anqiXu33/dialogsum-qlora)
- **Versions:** `main` / `v2-full` (this card, recommended); `v1-3k-1epoch` (earlier 3k-dialogue run)

## Results

DialogSum test set (500 dialogues, 3 reference summaries each; scores averaged
over references). Greedy decoding, same prompt for every model.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Words |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct, zero-shot | 0.227 | 0.062 | 0.174 | 0.879 | 41.0 |
| Qwen2.5-1.5B-Instruct, 1-shot | 0.241 | 0.075 | 0.192 | 0.884 | 33.7 |
| Qwen2.5-7B-Instruct, zero-shot | 0.288 | 0.085 | 0.219 | 0.879 | 39.6 |
| v1: this adapter, 3k dialogues | 0.389 | 0.141 | 0.313 | 0.912 | 22.7 |
| **v2: this adapter, 12.5k dialogues** | **0.414** | **0.157** | **0.335** | **0.917** | **21.9** |

Reference summaries average 18.8 words. v2 vs v1: +2.1 ROUGE-L, 95% CI
[+1.3, +3.0] (paired bootstrap over dialogues, p < 0.001). Even 500 training
dialogues (0.290 ROUGE-L) beat the 7B model zero-shot; see the GitHub repo for the
data-scaling curve and the loss-masking ablation.

In a manual check of 50 test dialogues, 72% of summaries were faithful and on
point; the main failure modes were unsupported details (16%) and confusing which
speaker did what (10%).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "AnqiXaq/dialogsum-qlora-adapter"   # add revision="v1-3k-1epoch" for v1

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)

dialogue = """#Person1#: Hi, I'd like to book a table for two tonight.
#Person2#: Sure, what time would you like?
#Person1#: Around 7 pm, if possible.
#Person2#: 7 pm works. Can I have your name?
#Person1#: It's Chen."""

messages = [
    {"role": "system", "content": "You read a conversation and produce a short structured summary."},
    {"role": "user", "content": "Summarise the conversation below. Return a topic label and a concise summary.\n\nConversation:\n" + dialogue},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Use this exact system and user prompt; the adapter was trained on it. To serve
without PEFT, merge the weights with `model.merge_and_unload()`.

## Training details

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct, 4-bit NF4, double quantisation |
| Method | QLoRA, TRL `SFTTrainer`, loss on the Topic/Summary answer only |
| LoRA | r = 16, α = 32, dropout 0.05, all attention and MLP projections (18.5M trainable parameters) |
| Data | All 12,460 DialogSum training dialogues, shuffled (seed 42) |
| Schedule | 1 epoch (1,557 steps), LR 2e-4 cosine, 20 warm-up steps, batch 4 × 2 accumulation, max length 1024 |
| Precision | fp16 compute, fp32 LoRA weights |
| Validation | Loss on 200 validation dialogues every 200 steps, best checkpoint kept (0.811) |
| Hardware | 1 × NVIDIA T4 (Kaggle), 85 minutes |

## Limitations

- English, everyday two-person conversations only; other domains (meetings,
  customer support, clinical dialogue) are untested.
- The model can attach a name to the wrong speaker or add a small unsupported
  detail. Review summaries before relying on them.
- One training run per configuration (seed 42).

## License

The adapter is released under Apache 2.0, matching the base model. DialogSum is
distributed under CC BY-NC-SA 4.0; check its terms before commercial use.

## Citation

```bibtex
@inproceedings{chen-etal-2021-dialogsum,
  title     = {{D}ialog{S}um: A Real-Life Scenario Dialogue Summarization Dataset},
  author    = {Chen, Yulong and Liu, Yang and Chen, Liang and Zhang, Yue},
  booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  year      = {2021}
}
```
