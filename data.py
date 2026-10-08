"""Prompt formatting shared by train / inference / eval.

Target is Topic + Summary rather than a plain paragraph. The topic comes straight
from DialogSum's existing `topic` column, so the structure costs no extra labelling.
"""

SYSTEM = "You read a conversation and produce a short structured summary."

USER_TEMPLATE = (
    "Summarise the conversation below. "
    "Return a topic label and a concise summary.\n\n"
    "Conversation:\n{dialogue}"
)


def format_target(topic, summary):
    return f"Topic: {topic}\nSummary: {summary}"


def build_messages(dialogue, topic=None, summary=None, shots=()):
    """Chat messages for the dialogue.

    shots: optional in-context examples (dicts with dialogue/topic/summary), inserted
           as user/assistant turns before the real one. Used only to give the *base*
           model a fair chance at the output format during evaluation.
    With topic+summary -> includes the assistant turn (training).
    Without -> stops after the user turn (inference).
    """
    messages = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(dialogue=s["dialogue"])})
        messages.append({"role": "assistant", "content": format_target(s["topic"], s["summary"])})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)})
    if topic is not None and summary is not None:
        messages.append({"role": "assistant", "content": format_target(topic, summary)})
    return messages


def to_prompt_completion(example):
    """Conversational prompt/completion format for TRL's SFTTrainer.

    With this format SFTTrainer computes the loss only on the completion
    (the assistant's Topic/Summary), not on the system prompt or the dialogue.
    """
    return {
        "prompt": build_messages(example["dialogue"]),
        "completion": [
            {"role": "assistant", "content": format_target(example["topic"], example["summary"])}
        ],
    }


def to_training_text(example, tokenizer):
    """Single flat text (v1 behaviour): loss is computed on every token, dialogue included."""
    messages = build_messages(example["dialogue"], example["topic"], example["summary"])
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def extract_summary(text):
    """Pull the summary out of a model output. Returns (summary, followed_format)."""
    if "Summary:" in text:
        return text.split("Summary:", 1)[1].strip(), True
    return text.strip(), False
