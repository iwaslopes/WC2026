"""World-Football-Elo ratings computed from the full results history.

Produces (a) the latest rating for every team (the strength feature used at
prediction time) and (b) a leak-free, match-level table of *pre-match* ratings
that the Poisson model trains on.
"""
import numpy as np
import pandas as pd
from . import config


def classify_importance(tournament: str) -> str:
    t = str(tournament).lower()
    if "friendly" in t:
        return "friendly"
    if "qualification" in t or "qualifier" in t:
        return "wc_qualifier" if "world cup" in t else "confed_qualifier"
    if "nations league" in t:
        return "nations_league"
    if "world cup" in t:
        return "world_cup"
    # continental final tournaments
    for k in ["euro", "copa am", "african cup", "afc asian", "gold cup",
              "confederations", "oceania nations", "uefa euro"]:
        if k in t:
            return "continental_final"
    return "other"


def _goal_mult(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0          # 3 -> 1.75, 4 -> 1.875, ...


def compute(results: pd.DataFrame):
    """results: columns date, home_team, away_team, home_score, away_score,
    tournament, neutral (bool). Returns (latest_elo dict, train_df)."""
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}
    rows = []
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        rh = elo.get(h, config.ELO_START)
        ra = elo.get(a, config.ELO_START)
        neutral = bool(getattr(r, "neutral", False))
        hfa = 0.0 if neutral else config.ELO_HOME_ADV

        # expected score for home
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + hfa)) / 400.0))
        gd = r.home_score - r.away_score
        res_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = config.ELO_K_BY_IMPORTANCE[classify_importance(r.tournament)] * _goal_mult(gd)
        delta = k * (res_h - exp_h)

        # record PRE-match ratings (leak-free) for model training
        rows.append((r.date, h, a, rh, ra, neutral,
                     int(r.home_score), int(r.away_score), r.tournament))

        elo[h] = rh + delta
        elo[a] = ra - delta

    train = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "elo_home_pre", "elo_away_pre",
        "neutral", "home_score", "away_score", "tournament"])
    return elo, train
