"""Gradio demo for AnqiXaq/dialogsum-qlora-adapter on a ZeroGPU Space."""
import spaces  # must be imported before torch on ZeroGPU

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "AnqiXaq/dialogsum-qlora-adapter"

SYSTEM = "You read a conversation and produce a short structured summary."
USER_TEMPLATE = (
    "Summarise the conversation below. Return a topic label and a concise summary.\n\n"
    "Conversation:\n{dialogue}"
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
# Load and merge on CPU: at startup ZeroGPU has no real GPU, and PEFT would try to
# put the adapter weights on cuda directly. Only the final .to("cuda") is emulated.
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER, torch_device="cpu")
model = model.merge_and_unload()  # plain model, faster inference
model.to("cuda")
model.eval()


@spaces.GPU(duration=30)
def summarise(dialogue: str) -> str:
    dialogue = (dialogue or "").strip()
    if not dialogue:
        return "Please paste a conversation."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)},
    ]
    # return_dict=True gives the same result on transformers 4.x and 5.x
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


EXAMPLES = [
    [
        "#Person1#: Hi, I'd like to book a dentist appointment, please.\n"
        "#Person2#: Sure. Is Tuesday OK for you?\n"
        "#Person1#: Tuesday is fine. Do you have anything in the afternoon?\n"
        "#Person2#: We have 3 PM available.\n"
        "#Person1#: Perfect, I'll take it."
    ],
    [
        "#Person1#: Did you finish the slides for tomorrow's client meeting?\n"
        "#Person2#: Almost. I still need the sales figures from last quarter.\n"
        "#Person1#: I can send them in an hour. Can you add a summary slide too?\n"
        "#Person2#: Sure, I'll put it at the end and send you the deck tonight."
    ],
]

demo = gr.Interface(
    fn=summarise,
    inputs=gr.Textbox(lines=12, label="Conversation", placeholder="#Person1#: ...\n#Person2#: ..."),
    outputs=gr.Textbox(lines=4, label="Topic and summary"),
    title="Dialogue Summariser (Qwen2.5-1.5B + QLoRA)",
    description=(
        "Paste an English conversation and get a topic label plus a short summary. "
        "Qwen2.5-1.5B-Instruct fine-tuned with QLoRA on DialogSum. "
        "[Model card](https://huggingface.co/AnqiXaq/dialogsum-qlora-adapter) · "
        "[Code](https://github.com/anqiXu33/dialogsum-qlora)"
    ),
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
