"""Paired bootstrap test between eval runs, using the per-example predictions.

    python analysis/significance.py results v2 v1 base-1shot base-0shot

Compares each run with the next one in the list (v2 vs v1, v1 vs base-1shot, ...).
ROUGE is computed per example with the same settings as `evaluate`'s rouge
(rouge_score, no stemming), so the means match run_eval.py.

The DialogSum test split has 500 dialogues with 3 reference summaries each
(ids test_<k>_1..3). Scores are averaged over the references of each dialogue
and the bootstrap resamples dialogues, not rows, so the three rows of one
dialogue are never treated as independent evidence.
"""

import json
import sys

import numpy as np
from rouge_score import rouge_scorer

B = 10_000
METRICS = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=False)


def per_example(path):
    """Per-dialogue scores: mean over that dialogue's reference summaries."""
    by_dialogue = {}
    for line in open(path):
        r = json.loads(line)
        s = scorer.score(r["reference"], r["summary"])
        d = by_dialogue.setdefault(r["id"].rsplit("_", 1)[0], {m: [] for m in METRICS})
        for m in METRICS:
            d[m].append(s[m].fmeasure)
    ids = sorted(by_dialogue)
    return ids, {m: np.array([np.mean(by_dialogue[i][m]) for i in ids]) for m in METRICS}


def main():
    out_dir, names = sys.argv[1], sys.argv[2:]
    data = {n: per_example(f"{out_dir}/{n}.jsonl") for n in names}
    rng = np.random.default_rng(0)
    n_ex = len(next(iter(data.values()))[0])
    idx = rng.integers(0, n_ex, size=(B, n_ex))

    for a, b in zip(names, names[1:]):
        assert data[a][0] == data[b][0], "runs must cover the same test examples"
        print(f"\n{a} vs {b}  ({n_ex} dialogues, {B:,} bootstrap resamples)")
        for m in METRICS:
            diff = data[a][1][m] - data[b][1][m]
            boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))  # two-sided
            wins = (diff > 0).mean()
            print(f"  {m:7s} {data[a][1][m].mean():.4f} vs {data[b][1][m].mean():.4f} | "
                  f"diff {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"p(two-sided)={p:.4f}  {a} better on {wins:.0%} of dialogues")


if __name__ == "__main__":
    main()
