"""Gradio Space demo. Paste a conversation, get the structured summary.

ZeroGPU differs from train.py:
  * fp16, not 4-bit. bitsandbytes is flaky on ZeroGPU and a 1.5B loads fine in fp16.
  * Model loads on CPU; the @spaces.GPU call moves it to cuda and grabs a GPU per request.
  * Space hardware = ZeroGPU, SDK = Gradio.

Push the adapter to the Hub first and point ADAPTER at that repo.
"""

import torch
import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "your-username/qlora-dialogsum-adapter"   # <-- real Hub repo

SYSTEM = "You read a conversation and produce a short structured summary."
USER_TEMPLATE = (
    "Summarise the conversation below. "
    "Return a topic label and a concise summary.\n\nConversation:\n{dialogue}"
)

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()


@spaces.GPU
def summarise(dialogue):
    model.to("cuda")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


example = (
    "#Person1#: Hi, I'd like to book a dentist appointment for next Tuesday.\n"
    "#Person2#: We have 3 PM on Tuesday available. Does that work?\n"
    "#Person1#: Yes, please put me down."
)

demo = gr.Interface(
    fn=summarise,
    inputs=gr.Textbox(lines=10, label="Conversation"),
    outputs=gr.Textbox(label="Structured summary"),
    title="DialogSum qLoRA — interaction summariser",
    description=(
        "A small Qwen2.5 fine-tuned with qLoRA to condense a multi-turn "
        "conversation into a topic label and a concise summary."
    ),
    examples=[[example]],
)

if __name__ == "__main__":
    demo.launch()
