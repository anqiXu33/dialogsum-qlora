"""Paired bootstrap test between eval runs, using the per-example predictions.

    python analysis/significance.py results v2 v1 base-1shot base-0shot

Compares each run with the next one in the list (v2 vs v1, v1 vs base-1shot, ...).
ROUGE is computed per example with the same settings as `evaluate`'s rouge
(rouge_score, no stemming), so the means match run_eval.py.
"""

import json
import sys

import numpy as np
from rouge_score import rouge_scorer

B = 10_000
METRICS = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=False)


def per_example(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["id"])
    scores = {m: [] for m in METRICS}
    for r in rows:
        s = scorer.score(r["reference"], r["summary"])
        for m in METRICS:
            scores[m].append(s[m].fmeasure)
    return [r["id"] for r in rows], {m: np.array(v) for m, v in scores.items()}


def main():
    out_dir, names = sys.argv[1], sys.argv[2:]
    data = {n: per_example(f"{out_dir}/{n}.jsonl") for n in names}
    rng = np.random.default_rng(0)
    n_ex = len(next(iter(data.values()))[0])
    idx = rng.integers(0, n_ex, size=(B, n_ex))

    for a, b in zip(names, names[1:]):
        assert data[a][0] == data[b][0], "runs must cover the same test examples"
        print(f"\n{a} vs {b}  (n={n_ex}, {B:,} bootstrap resamples)")
        for m in METRICS:
            diff = data[a][1][m] - data[b][1][m]
            boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = (boots <= 0).mean() if diff.mean() > 0 else (boots >= 0).mean()
            wins = (diff > 0).mean()
            print(f"  {m:7s} {data[a][1][m].mean():.4f} vs {data[b][1][m].mean():.4f} | "
                  f"diff {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"p={p:.4f}  {a} better on {wins:.0%} of dialogues")


if __name__ == "__main__":
    main()
