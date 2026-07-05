"""Score the model on DialogSum's test split: ROUGE + BERTScore.

Run with USE_ADAPTER=False, then True, to see the delta from the adapter.
"""

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data import build_messages

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "qlora-dialogsum-adapter"
USE_ADAPTER = True     # False -> score the base model
N = 100

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto"
)
if USE_ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def generate(dialogue):
    prompt = tok.apply_chat_template(
        build_messages(dialogue), tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "Summary:" in text:                       # drop the scaffold, score the summary
        text = text.split("Summary:", 1)[1]
    return text.strip()


def main():
    ds = load_dataset("knkarthick/dialogsum", split="test").select(range(N))
    preds = [generate(x["dialogue"]) for x in ds]
    refs = [x["summary"] for x in ds]

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    print("ROUGE:", rouge.compute(predictions=preds, references=refs))
    bs = bertscore.compute(predictions=preds, references=refs, lang="en")
    print("BERTScore F1 (mean):", sum(bs["f1"]) / len(bs["f1"]))


if __name__ == "__main__":
    main()
