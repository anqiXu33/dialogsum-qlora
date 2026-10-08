"""Data-scaling figure: ROUGE-L vs number of training dialogues.

    python analysis/plot_scaling.py            # writes assets/scaling.png
"""

import json
import os

import matplotlib.pyplot as plt

RESULTS = "results"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = "#2a78d6"

points = [(500, "n500"), (1000, "n1000"), (3000, "n3000"), (12460, "v2")]
refs = [("qwen7b-0shot", "Qwen2.5-7B, zero-shot"), ("base-1shot", "Qwen2.5-1.5B, 1-shot"),
        ("base-0shot", "Qwen2.5-1.5B, zero-shot")]


def rouge_l(name):
    return json.load(open(os.path.join(RESULTS, f"{name}.json")))["rougeL"]


xs = [n for n, _ in points]
ys = [rouge_l(name) for _, name in points]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for name, label in refs:
    y = rouge_l(name)
    ax.axhline(y, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(58000, y + 0.002, f"{label}  {y:.3f}", color=INK2, fontsize=8.5, va="bottom", ha="right")

ax.plot(xs, ys, color=SERIES, lw=2, zorder=3)
ax.scatter(xs, ys, s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=4)
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=INK)
ax.text(xs[-1] * 1.08, ys[-1], "QLoRA fine-tuned 1.5B", color=INK, fontsize=9,
        va="center", ha="left", fontweight="bold")

ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["500", "1k", "3k", "12.5k\n(full)"])
ax.minorticks_off()
ax.set_xlim(380, 60000)
ax.set_ylim(0.15, 0.36)
ax.set_xlabel("Training dialogues (log scale)", color=INK2, fontsize=9.5)
ax.set_ylabel("ROUGE-L (DialogSum test, 500 dialogues)", color=INK2, fontsize=9.5)
ax.set_title("Fine-tuning a 1.5B model on 500 dialogues beats a 7B model zero-shot",
             color=INK, fontsize=11, loc="left", pad=12)

ax.grid(axis="y", color=GRID, lw=0.8)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)

os.makedirs("assets", exist_ok=True)
fig.tight_layout()
fig.savefig("assets/scaling.png", facecolor=SURFACE)
print("wrote assets/scaling.png")
