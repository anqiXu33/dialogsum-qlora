"""Print a Markdown table of every results/*.json produced by run_eval.py."""

import glob
import json
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = [json.load(open(p)) for p in sorted(glob.glob(f"{out_dir}/*.json"))]
if not rows:
    sys.exit(f"No results in {out_dir}/")

cols = [("name", "Run"), ("n", "N"), ("format_followed", "Format %"), ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore F1"),
        ("mean_pred_words", "Pred words")]
print("| " + " | ".join(c[1] for c in cols) + " |")
print("|" + "---|" * len(cols))
for r in sorted(rows, key=lambda r: r["rougeL"]):
    cells = []
    for key, _ in cols:
        v = r.get(key)
        if key == "format_followed":
            cells.append(f"{100 * v:.0f}%")
        elif isinstance(v, float):
            cells.append(f"{v:.1f}" if key == "mean_pred_words" else f"{v:.3f}")
        else:
            cells.append(str(v))
    print("| " + " | ".join(cells) + " |")
