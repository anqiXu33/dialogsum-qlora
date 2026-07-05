"""Load base + adapter and summarise a single conversation. Quick post-train check."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data import build_messages

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "qlora-dialogsum-adapter"   # local folder or Hub repo id

COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()


def summarise(dialogue, max_new_tokens=128):
    messages = build_messages(dialogue)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]   # only the generated part
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    demo = (
        "#Person1#: Hi, I'd like to book a dentist appointment for next Tuesday.\n"
        "#Person2#: We have 3 PM on Tuesday available. Does that work?\n"
        "#Person1#: Yes, please put me down."
    )
    print(summarise(demo))
