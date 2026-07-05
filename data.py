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


def build_messages(dialogue, topic=None, summary=None):
    """Chat messages for the dialogue.

    With topic+summary -> includes the assistant turn (training).
    Without -> stops after the user turn (inference).
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)},
    ]
    if topic is not None and summary is not None:
        messages.append(
            {"role": "assistant",
             "content": f"Topic: {topic}\nSummary: {summary}"}
        )
    return messages


def to_training_text(example, tokenizer):
    messages = build_messages(
        example["dialogue"], example["topic"], example["summary"]
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}
