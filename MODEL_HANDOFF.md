# WC2026 model — handoff / resume

## ⭐ UPDATE (session 2) — learned weights + live results
The hand-set strength blend (W_FORM / W_TOUR / W_PED) was **replaced by a model that
learns each variable's weight** from history. The goals model is one symmetric
Poisson over leak-free, as-of features; current **learned importance**:

| Variable | Learned weight | Notes |
|---|---|---|
| Recent form | ~40% | recency-weighted last ~20 matches (priority 1) |
| FIFA ranking + Elo | ~39% | team quality / pedigree |
| Head-to-head vs FIFA-rank (tier) | ~13% | record vs the current opponent's calibre (priority 2) |
| Major-tournament form | ~8% | World Cups / continental finals (priority 3) |

Also new this session:
- **Already-played games** are read from `data/wc2026_results_so_far.csv`, locked to
  their real scores in the group output, and **condition the knockout simulation**.
- **Group ties** are now predicted (`GROUP_TIE_THRESHOLD`=0.16) → ~28% draw rate
  (previously 0). Played group games already include ~7 draws.
- **Home advantage fully removed** (neutral, one symmetric model, one MAE = 0.958).
- Placeholder play-off slots show **real team names** (Czechia, Türkiye, Sweden, Iraq, etc.).
- Hand-set knobs now superseded by learned weights: `W_FORM`, `W_TOUR`, `RATING_GAIN`,
  `TIER_ELO_PER_PPG` are **no longer used** by the goals model (kept in file but inert).

Current headline: validation goal MAE **0.958**, exact 12.4%; title odds
**Spain 25%, Argentina 20%, France 13%, England 6%, Brazil 5%**; ~20/72 group ties.

Active folder: `wc2026_snapshot_20260608/`. Engine: `wc2026/integrated.py`
(note: `wc2026/learned_weights.py` is a separate experiment, not used by the notebook).

---


State as we left it. Engine: `wc2026/integrated.py` (numpy + pandas only).
Run: `from wc2026 import integrated; eng, group_predictions, knockout_predictions = integrated.run_all(n_sims=20000)`
or run `notebook.ipynb` top to bottom. Outputs land in `predictions/`.

## Headline output (this version)
- Validation: **goal MAE 0.96**, exact-score 12.6% (hold-out from 2024-09-09).
- Title odds: **Spain 26%, Argentina 17%, France 10%, Belgium 7%, Portugal 7%, England 6%, Germany 6%, Morocco 5%**.
- Predicted final: **Spain vs Argentina → Spain**. Avg group goals/match 1.76; corners 6–12; yellows 2–7; ~13 reds/tournament; ~6 KO shootouts.

## Data sources
- `data/external/international_results.csv` — 49k internationals (results, recency, form, tier, tournament metrics).
- `data/processed/team_strength.csv` — per-team `custom_elo`, `fifa_points` (EA FC columns exist but are NOT used).
- `data/processed/training_matches.csv` — engineered rows (elo, recent form, importance, sample_weight) the goals model trains on.
- `data/group_fixtures.csv`, `data/knockout_slots.csv` — the competition fixtures/bracket.
- `config/placeholder_teams.csv` — play-off slots resolved to qualified teams. `config/team_set_pieces.csv` — per-team corner/yellow medians (editable, auto-written).

## Pipeline flow
1. Build a **strength rating** per team (4 components, below).
2. Fit a **single symmetric Poisson goals model** (team-perspective; one MAE, no home/away).
3. For each matchup, apply **matchup-level adjustments** (opponent-tier, WC-goals, rating-gain) → expected goals (λ).
4. Pick scoreline that **maximises expected rubric points**; derive winner/penalties.
5. **Monte-Carlo** (20,000 sims) the whole bracket for knockout matchups; predict each KO match on the modal matchup.

## Strength rating = blend of 4 components
`rating = W_PED·pedigree + W_FORM·form + W_TOUR·tournament`  (W_PED = 1 − W_FORM − W_TOUR = 0.40)

| Component | Weight (current) | What it is |
|---|---|---|
| Pedigree (FIFA + Elo) | **0.40** | `W_FIFA`=0.60 split: 0.60 FIFA points + 0.40 results-Elo, mapped to one scale |
| Recent form | **W_FORM = 0.40** (main driver) | recency-weighted ppg + 0.3·GD over last `FORM_N`=20 matches, half-life `FORM_HL_DAYS`=400d |
| Main-competition | **W_TOUR = 0.20** | recency-weighted ppg + 0.3·GD in WC + continental finals (`TOUR_N`=30, half-life `TOUR_HL_DAYS`=4y) |

Then two **matchup-level** adjustments (not per-team, computed per game):
- **Opponent-tier** (`TIER_ELO_PER_PPG`=90): shifts a team's effective rating by how it has done vs opponents of the *current opponent's* calibre (within `TIER_BAND`=120 elo), shrunk toward a low prior (`TIER_PRIOR_PPG`=1.25, `TIER_K`=4) when untested. This is the "Morocco is great vs weak, weak vs elite" fix.
- **WC-goals attack nudge** (`WC_GOAL_WEIGHT`=0.10): small ±8% multiplier on a team's λ from its goals/game in World Cups since `WC_GOALS_SINCE`=2006.
- **`RATING_GAIN`=1.35**: amplifies the favourite/underdog gap (more discrepancy).

## Goals model
Single symmetric Poisson (own goals ~ rating_diff, rating_sum, own recent gf/ga/win-rate, opp recent gf/ga/win-rate, importance). Pure-numpy ridge IRLS (`alpha`=0.03). **Neutral venue** — no home advantage anywhere (each team's λ comes from the same model applied from its side).
Training sample weights = `recency × phase × ko_proxy`:
- recency half-life `RECENCY_HALFLIFE_DAYS` = 1.5y (steep: ~60% of weight in last 2 years).
- phase (`phase_weight`): World Cup 5, continental finals 4, Nations League 3, qualifiers 3, friendly 1, other 2.
- `KO_PROXY_BOOST`=3.0 for neutral games between two sides both above `KO_PROXY_ELO`=1700 (mirrors KO conditions).

## Scoreline / winner / penalties
- Group & KO scorelines maximise **expected rubric points** (exact 25; correct GD or total 10; group winner 40).
- KO penalties: a near coin-flip tie (|P(home)−P(away)| < `KO_TIE_THRESHOLD`=0.12) is predicted as a draw → `penalties = True`; otherwise decisive. Penalties always equal "predicted score is a draw."
- KO bracket is propagated consistently (R32 from sim; later rounds filled by predicted winner/loser of feeder match → third-place game = the two semifinal losers).

## Corners / yellows / reds (heuristics — no card data in source)
- Per-team medians from team strength, summed per matchup + seeded jitter (reproducible via `[SEED, match_id]`).
- Corners: `CORNER_PER_TEAM_BASE`=4.4 + `CORNER_RATING_COEF`=0.55·z(rating). Yellows: `YELLOW_PER_TEAM_BASE`=1.9 − `YELLOW_RATING_COEF`=0.20·z (underdogs foul more) + KO bump.
- Reds: low per-match probability `RED_BASE`=0.055 (+`RED_KO_BONUS`=0.04 in KO) → ~13/tournament.
- Per-team values written to `config/team_set_pieces.csv` (editable with real stats).

## All tunable knobs (top of `integrated.py`)
`W_FIFA`=0.60, `W_FORM`=0.40, `W_TOUR`=0.20, `FORM_N`=20, `FORM_HL_DAYS`=400, `FORM_RATES_N`=12,
`TOUR_N`=30, `TOUR_HL_DAYS`=4y, `MAIN_COMP_MIN_PHASE`=4, `WC_GOAL_WEIGHT`=0.10, `WC_GOALS_SINCE`=2006,
`TIER_ELO_PER_PPG`=90, `TIER_BAND`=120, `TIER_HIST_N`=40, `TIER_PRIOR_PPG`=1.25, `TIER_K`=4, `TIER_ADJ_CLIP`=1.5,
`RECENCY_HALFLIFE_DAYS`=1.5y, `RATING_GAIN`=1.35, `MAX_GOALS`=8, `KO_TIE_THRESHOLD`=0.12,
`KO_PROXY_ELO`=1700, `KO_PROXY_BOOST`=3.0, `RED_BASE`=0.055, `RED_KO_BONUS`=0.04, `N_SIMS`=20000, `SEED`=42,
`CORNER_*`, `YELLOW_*`.

## Known caveats / quirks (for next session)
- Croatia scores low on the tournament component because their deep runs come via penalty shootouts (count as regulation draws → low ppg).
- The tournament component counts Gold Cup / Asian Cup as "main," which can flatter CONCACAF/AFC sides (e.g. Mexico) — the opponent-tier adjustment partly offsets this.
- Opponent-tier uses *current* ratings for historical opponents (approximation; recency-weighted so recent ratings dominate).
- "exact score" is hard (~12%); the model deliberately hedges scorelines for expected points, not realism.

## Session changelog (what we changed & why)
1. Integrated the EA FC group pipeline + added the missing knockout Monte-Carlo bracket.
2. Dropped EA FC ratings → strength = FIFA + Elo.
3. Steep recency discount; recent qualifiers/Nations League weighted up.
4. Fixed corners (per-team medians summed, recentred ~10); varying yellows; added occasional reds.
5. Fixed KO penalties (drawn result → penalties) and the third-place bug (consistent bracket).
6. Scrapped home advantage (single symmetric goals model, one MAE).
7. Added recent-form component to the rating (Brazil marked down).
8. Added small WC-goals attacking nudge.
9. Added opponent-tier adjustment (Morocco strong vs weak, weak vs elite).
10. Added main-competition (tournament pedigree) component, weighted below recent form.

## Ideas to discuss next time
- Re-weight the tournament component to discount weaker continental cups (Gold Cup vs Euro).
- Handle penalty-shootout teams (Croatia) so deep-run pedigree isn't undercounted.
- Calibrate `RATING_GAIN` / `KO_TIE_THRESHOLD` against historical WC outcomes.
- Optionally fold opponent-tier form into the goals-model training (currently a rating-side adjustment).
