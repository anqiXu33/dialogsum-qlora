"""Score a model on DialogSum's test split: ROUGE + BERTScore.

Every run writes two files to results/:
    <name>.jsonl   one line per test example (dialogue id, reference, raw output, summary)
    <name>.json    aggregate metrics + how often the output followed the Topic/Summary format

Examples
--------
    python run_eval.py --name base                       # untuned base model, zero-shot
    python run_eval.py --name base-1shot --shots 1       # base with one in-context example
    python run_eval.py --name v2 --adapter runs/v2/adapter
    python run_eval.py --name v1 --adapter AnqiXaq/dialogsum-qlora-adapter --revision v1-3k-1epoch
    python compare.py                                    # table of everything in results/
"""

import argparse
import json
import os
import time

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data import build_messages, extract_summary

DATASET_ID = "knkarthick/dialogsum"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, used for the output files")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="local folder or Hub repo id; omit for the base model")
    p.add_argument("--revision", default=None, help="Hub tag/branch of the adapter, e.g. v1-3k-1epoch")
    p.add_argument("--n", type=int, default=None, help="first N test examples (default: all 1,500)")
    p.add_argument("--shots", type=int, default=0, help="in-context examples from the train split")
    p.add_argument("--no-4bit", action="store_true", help="load the model in bf16/fp16 instead of 4-bit")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def load_model(args):
    # bf16 only on Ampere+; T4 reports bf16 support but emulates it slowly
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    kwargs = {"device_map": {"": 0}, "dtype": dtype}
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, revision=args.revision)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # required for batched generation with a decoder-only model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args)

    test = load_dataset(DATASET_ID, split="test")
    if args.n:
        test = test.select(range(args.n))
    shots = []
    if args.shots:
        shots = list(load_dataset(DATASET_ID, split="train").select(range(args.shots)))

    prompts = [
        tok.apply_chat_template(build_messages(x["dialogue"], shots=shots),
                                tokenize=False, add_generation_prompt=True)
        for x in test
    ]

    raw_outputs = []
    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, batch["input_ids"].shape[1]:]
        raw_outputs += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"{min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    gen_seconds = time.time() - t0

    preds, followed = zip(*(extract_summary(o) for o in raw_outputs))
    refs = [x["summary"] for x in test]

    rouge = evaluate.load("rouge").compute(predictions=list(preds), references=refs)
    bs = evaluate.load("bertscore").compute(predictions=list(preds), references=refs, lang="en")

    metrics = {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "revision": args.revision,
        "shots": args.shots,
        "n": len(test),
        "format_followed": sum(followed) / len(followed),
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
        "bertscore_f1": sum(bs["f1"]) / len(bs["f1"]),
        "mean_pred_words": sum(len(p.split()) for p in preds) / len(preds),
        "mean_ref_words": sum(len(r.split()) for r in refs) / len(refs),
        "generation_seconds": round(gen_seconds, 1),
    }

    with open(os.path.join(args.out_dir, f"{args.name}.jsonl"), "w") as f:
        for x, raw, pred, ok in zip(test, raw_outputs, preds, followed):
            f.write(json.dumps({"id": x["id"], "topic_ref": x["topic"], "reference": x["summary"],
                                "raw_output": raw, "summary": pred, "followed_format": ok}) + "\n")
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
