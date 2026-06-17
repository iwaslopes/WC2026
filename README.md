# World Cup 2026 predictions

Fills `notebook.ipynb` with predictions for all 104 matches, optimised against the
competition scoring rubric. Runs on numpy + pandas only.

## Run it
```python
from wc2026 import integrated
eng, group_predictions, knockout_predictions = integrated.run_all(n_sims=20000)
```
or just run `notebook.ipynb` top to bottom.

## Approach (`wc2026/integrated.py`)
- **Strength** = **FIFA ranking points** (primary) blended with a results-based **Elo**
  (49k international matches). EA FC ratings are intentionally NOT used.
- **Goals model** = Poisson regression on the engineered match history, with sample
  weights that strongly favour **finals tournaments** (World Cups, Euros, Copa, AFCON…)
  over qualifiers and friendlies — so tournament matchups drive the prediction.
  Validation: home/away goal MAE ≈ 1.05 / 0.86, exact-score ≈ 12%.
- **Scorelines** maximise **expected rubric points** (exact 25; goal-diff/total 10; group winner 40).
- **Knockouts** = 20,000-run Monte-Carlo of the full bracket → most-likely matchup per slot.
  A near coin-flip tie (|P(home)−P(away)| < `KO_TIE_THRESHOLD`, default 0.08) is predicted
  as a draw and **goes to penalties**; clear games get a decisive score. Score, winner and
  penalties are mutually consistent.
- **Corners & yellow cards** vary by match (more corners with more expected goals; more
  yellows in tight, high-stakes, knockout games). No card data exists in the source, so
  these are principled heuristics, not fitted.

## Tunable knobs (top of `integrated.py`)
- `W_FIFA` (default 0.60) — FIFA-vs-Elo blend in the strength rating.
- `phase_weight()` — how much each competition type counts in the fit.
- `KO_TIE_THRESHOLD` — how close a tie must be to be sent to penalties.
- `RECENCY_HALFLIFE_DAYS`, `N_SIMS`.

## Notes
- Active engine is `wc2026/integrated.py`. Other `wc2026/*.py` files are an earlier
  standalone variant kept as reference — safe to delete. `data/results.csv` is a leftover
  synthetic test file — safe to delete. The original group-only pipeline (needs scikit-learn)
  is `src/wc2026_pipeline.py`; this engine reproduces and extends it.
