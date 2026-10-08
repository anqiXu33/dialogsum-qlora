# Analysis

- `significance.py`: paired bootstrap between eval runs. Resamples the 500 test
  dialogues (each has 3 reference summaries), two-sided p-values.
  `python analysis/significance.py results v2 v1`
- `plot_scaling.py`: data-scaling figure in `assets/`.
- `error_annotations.csv`: manual error analysis of v2 on 50 test dialogues
  sampled with `random.seed(0)` (one annotator). Labels:
  - `ok`: faithful and covers the main point (minor omissions allowed)
  - `speaker`: who-did-what confusion (wrong speaker, swapped roles)
  - `factual`: a detail or causal link not supported by the dialogue
  - `missed_point`: faithful details but misses what the conversation is about

  Source dialogues are from the original DialogSum release
  (github.com/cylnlp/dialogsum, `dialogsum.test.jsonl`).
