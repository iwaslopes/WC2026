# FIFA World Cup 2026 — Match Predictor

A statistical model that predicts **every match of the 2026 FIFA World Cup** — all 104 games
across the group stage and knockout rounds — producing **goals (scorelines), corners,
yellow cards and red cards**, plus group winners, knockout matchups, match winners and
penalty shootouts.

Built for the **DataCamp "Predict the FIFA World Cup 2026" competition**, where predictions
must be submitted for every fixture before a ball is kicked and are graded against a points
rubric (exact scoreline, goal difference / total goals, corners, cards, winners, matchups
and penalties, with later rounds carrying score multipliers).

Runs on **numpy + pandas only** 

## Quick start

```python
from wc2026 import integrated

eng, group_predictions, knockout_predictions = integrated.run_all(n_sims=20000)
group_predictions.to_csv("predictions/group_predictions.csv", index=False)
knockout_predictions.to_csv("predictions/knockout_predictions.csv", index=False)
```

or just open **`notebook.ipynb`** and run it top to bottom — it fills the competition
submission tables for the group and knockout stages.

## What it predicts

| Output | How |
|---|---|
| **Scoreline** (goals) | Poisson goals model → scoreline that maximises expected rubric points |
| **Group winner** (home/draw/away) | from the goals model; realistic draw rate (~28%) |
| **Knockout matchups** | 20,000-run Monte-Carlo simulation of the whole bracket |
| **Match winner & penalties** | drawn knockout result → goes to a shootout |
| **Corners / yellow / red cards** | per-team rates (from team strength) summed per match + variation |

## How the model works (`wc2026/integrated.py`)

A single **symmetric Poisson goals model** — neutral venue, no home/away bias (World Cup
games are played on neutral ground). It **learns the weight of each variable** from ~49k
historical internationals via Poisson regression. The learned importance:

| Variable | Learned weight | What it captures |
|---|---|---|
| **Recent form** | ~43% | recency-weighted goals & results over each team's last ~20 matches |
| **FIFA ranking + Elo** | ~39% | team quality / pedigree |
| **Head-to-head vs FIFA rank** | ~10% | record vs opponents of the *current* opponent's calibre |
| **Major-tournament form** | ~8% | performance in World Cups / continental finals vs Elo expectation |

All features are computed **leak-free** (only information available before each match) with a
**steep time discount** (recent results dominate; a game from 10 years ago counts ~1%).

- **Scorelines** are chosen to maximise **expected competition points** (exact 25; correct
  goal-difference or total 10; group winner 40), not just the single likeliest score.
- **Knockouts**: a 20,000-run Monte-Carlo of the bracket gives each slot's most-likely
  matchup; a coin-flip tie is predicted as a draw and **goes to penalties**.
- **Live results**: games already played (`data/wc2026_results_so_far.csv`) are locked to
  their real scores and condition the knockout simulation.
- **Corners / cards**: each team has its own per-game rate (in `config/team_set_pieces.csv`,
  editable); a match sums the two teams' rates plus seeded, reproducible variation. Red
  cards occur with a low per-match probability (uncommon, but they happen). The source has
  no corner/card data, so these are principled heuristics rather than fitted values.

**Validation:** goal MAE ≈ **0.87 on recent major-tournament matches** (Euro 2024,
Copa América 2024, AFCON, WC 2022 — the games most like the World Cup), and ≈ 0.95 across
*all* recent internationals. Tournament error is lower mainly because those games are
lower-scoring (fewer goals → smaller absolute errors), not because the model is sharper
there. Either way it's close to the irreducible noise floor of football. Exact-score ≈ 12%.

## Repository layout

```
notebook.ipynb              # competition submission notebook
wc2026/integrated.py        # the model engine (everything lives here)
config/
  placeholder_teams.csv     # play-off slots → qualified teams
  team_set_pieces.csv       # editable per-team corner / yellow-card rates
data/
  group_fixtures.csv        # 72 group matches
  knockout_slots.csv        # 32 knockout slots / bracket structure
  wc2026_results_so_far.csv # games already played (locked to actuals)
  external/                 # international results, FIFA rankings (model inputs)
  processed/                # engineered strength & training tables
predictions/                # output CSVs (group + knockout)
MODEL_HANDOFF.md            # detailed design notes & changelog
```

## Tunable knobs (top of `wc2026/integrated.py`)

The variable *weights* are learned, but the modelling choices are configurable:

- `W_FIFA` — FIFA-vs-Elo split inside the strength rating
- `RECENCY_HALFLIFE_DAYS` — how fast old results are discounted
- `GROUP_TIE_THRESHOLD` — how close a group game must be to be called a draw
- `KO_TIE_THRESHOLD` — how close a knockout tie must be to go to penalties
- `phase_weight()` — relative weight of each competition type in training
- `N_SIMS` — Monte-Carlo simulations (default 20,000)

## Notes

- The active engine is **`wc2026/integrated.py`**. Other `wc2026/*.py` files are earlier
  standalone experiments kept for reference.
- Method choices (neutral venue, Poisson over neural nets, opponent-quality adjustment) are
  documented with the reasoning in `MODEL_HANDOFF.md`.
- This uploaded version includes tweaks to the initial model. To improve it for the remaining games, the first 18 results of the 2026 WC were added.

## Tech stack

Python · numpy · pandas. Statistical model (Dixon–Coles-style Poisson) + Monte-Carlo simulation.
