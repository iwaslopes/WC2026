from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import scoring
from .integrated import (
    CORNER_PER_TEAM_BASE,
    CORNER_RATING_COEF,
    Engine as PreviousEngine,
    KO_TIE_THRESHOLD,
    RED_BASE,
    RED_KO_BONUS,
    SEED,
    YELLOW_PER_TEAM_BASE,
    YELLOW_RATING_COEF,
    team_key,
)


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "historical"
OUT = ROOT / "models"

FULL_RESULTS = ROOT / "data" / "external" / "international_results.csv"
WC_TARGETS = HIST / "worldcup_2014_2018_2022_matches_with_pre_tournament_rankings.csv"
CURRENT_RANKING = ROOT / "data" / "processed" / "fifa_mens_ranking.csv"
GROUP_FIXTURES = ROOT / "data" / "group_fixtures.csv"
KNOCKOUT_SLOTS = ROOT / "data" / "knockout_slots.csv"
PREDICTIONS = ROOT / "predictions"

TOURNAMENT_STARTS = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
    2026: pd.Timestamp("2026-06-11"),
}

QUALIFYING_STARTS = {
    2014: pd.Timestamp("2011-06-15"),
    2018: pd.Timestamp("2015-03-12"),
    2022: pd.Timestamp("2019-06-06"),
    2026: pd.Timestamp("2023-09-07"),
}

HOSTS = {
    2014: {"brazil"},
    2018: {"russia"},
    2022: {"qatar"},
    2026: {"usa", "canada", "mexico"},
}

MAX_GOALS = 8
RECENT_N = 14
SHORT_RECENT_N = 8
RECENT_HALFLIFE_DAYS = 220.0
SHORT_RECENT_HALFLIFE_DAYS = 120.0
CYCLE_HALFLIFE_DAYS = 365.0
PREVIOUS_WC_HALFLIFE_DAYS = 720.0
ADVERSARY_QUALITY_WEIGHT = 0.45
ELITE_ADVERSARY_THRESHOLD = 0.75
ELITE_RESULT_PRIOR_PPG = 1.15
ELITE_RESULT_PRIOR_GD = 0.0
FIFA_TIER_RECENT_N = 30
FIFA_TIER_HALFLIFE_DAYS = 260.0
FIFA_TIER_POINTS_HALFLIFE = 120.0
LEARNED_GOALS_WEIGHT = 0.45
PREVIOUS_ENGINE_GOALS_WEIGHT = 0.25
FIFA_TIER_DIRECT_GOALS_WEIGHT = 0.30
TOURNAMENT_CONTEXT_WEIGHT = 0.06
CURRENT_FORM_CONTEXT_WEIGHT = 0.42
CURRENT_FORM_CONTEXT_CLIP = 0.35
MISMATCH_EDGE_SCALE = 2.2
MISMATCH_TOTAL_GOAL_LIFT = 0.34
MISMATCH_FAVORITE_GOAL_LIFT = 0.78
MISMATCH_UNDERDOG_GOAL_TRIM = 0.12
PEDIGREE_ANCHOR_WEIGHT = 0.22
PEDIGREE_EDGE_SCALE = 150.0
RECENT_TIE_BREAK_EDGE = 0.05
KNOCKOUT_DRAW_EDGE_THRESHOLD = 0.035
WC_SAMPLE_RECENCY_WEIGHT = {
    2014: 0.55,
    2018: 0.90,
    2022: 1.35,
}
_QUALITY_CACHE: dict[tuple[int, str, int], dict[str, float]] = {}
_FIFA_POINTS_CACHE: dict[int, dict[str, float]] = {}

FEATURE_COLUMNS = [
    "fifa_points_diff",
    "fifa_rank_advantage",
    "recent_ppg_diff",
    "recent_gd_diff",
    "short_ppg_diff",
    "short_gd_diff",
    "own_recent_ppg",
    "own_recent_gf",
    "own_recent_ga",
    "own_recent_gd",
    "own_short_ppg",
    "own_short_gf",
    "own_short_ga",
    "own_short_gd",
    "own_recent_quality_ppg",
    "own_recent_quality_gd",
    "own_recent_adversary_quality",
    "own_elite_recent_ppg",
    "own_elite_recent_gd",
    "own_elite_recent_matches",
    "opp_recent_ppg",
    "opp_recent_gf",
    "opp_recent_ga",
    "opp_recent_gd",
    "opp_short_ppg",
    "opp_short_gf",
    "opp_short_ga",
    "opp_short_gd",
    "opp_recent_quality_ppg",
    "opp_recent_quality_gd",
    "opp_recent_adversary_quality",
    "opp_elite_recent_ppg",
    "opp_elite_recent_gd",
    "opp_elite_recent_matches",
    "quality_recent_ppg_diff",
    "quality_recent_gd_diff",
    "adversary_quality_diff",
    "elite_recent_ppg_diff",
    "elite_recent_gd_diff",
    "own_fifa_tier_gf",
    "own_fifa_tier_ga",
    "own_fifa_tier_gd",
    "own_fifa_tier_support",
    "opp_fifa_tier_gf",
    "opp_fifa_tier_ga",
    "opp_fifa_tier_gd",
    "opp_fifa_tier_support",
    "fifa_tier_attack_edge",
    "fifa_tier_defense_edge",
    "own_cycle_ppg",
    "own_cycle_gd",
    "opp_cycle_ppg",
    "opp_cycle_gd",
    "host_diff",
    "is_knockout",
]


@dataclass
class LearnedModels:
    goal_model: Pipeline
    outcome_model: Pipeline
    feature_columns: list[str]


def load_full_results() -> pd.DataFrame:
    df = pd.read_csv(FULL_RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home_key"] = df["home_team"].map(team_key)
    df["away_key"] = df["away_team"].map(team_key)
    return df.sort_values("date").reset_index(drop=True)


def load_worldcup_targets() -> pd.DataFrame:
    df = pd.read_csv(WC_TARGETS, parse_dates=["date"])
    df = df.sort_values(["world_cup", "date"]).reset_index(drop=True)
    df["match_order_in_wc"] = df.groupby("world_cup").cumcount()
    df["is_knockout"] = (df["match_order_in_wc"] >= 48).astype(int)
    return df


def _team_slice(results: pd.DataFrame, key: str) -> pd.DataFrame:
    return results[(results["home_key"] == key) | (results["away_key"] == key)].copy()


def _fifa_points_map(world_cup: int) -> dict[str, float]:
    if world_cup in _FIFA_POINTS_CACHE:
        return _FIFA_POINTS_CACHE[world_cup]

    if world_cup == 2026:
        ranking = pd.read_csv(CURRENT_RANKING)
        out = dict(zip(ranking["team_key"], ranking["points"].astype(float)))
    else:
        rankings = pd.read_csv(HIST / "fifa_rankings_worldcup_snapshots.csv")
        rankings = rankings[(rankings["world_cup"] == world_cup) & (rankings["usage"] == "pre_tournament")]
        out = dict(zip(rankings["team_key"], rankings["points"].astype(float)))
    _FIFA_POINTS_CACHE[world_cup] = out
    return out


def _summarize_team_matches(
    matches: pd.DataFrame,
    key: str,
    as_of: pd.Timestamp,
    half_life_days: float = RECENT_HALFLIFE_DAYS,
    adversary_quality: dict[str, float] | None = None,
) -> dict[str, float]:
    if matches.empty:
        return {
            "matches": 0.0,
            "ppg": 1.0,
            "gf": 1.25,
            "ga": 1.25,
            "gd": 0.0,
            "win_rate": 1 / 3,
            "quality_ppg": 1.0,
            "quality_gd": 0.0,
            "adversary_quality": 0.0,
        }

    is_home = matches["home_key"].eq(key).to_numpy()
    gf = np.where(is_home, matches["home_score"].to_numpy(), matches["away_score"].to_numpy())
    ga = np.where(is_home, matches["away_score"].to_numpy(), matches["home_score"].to_numpy())
    pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    dates = pd.to_datetime(matches["date"])
    weights = 0.5 ** ((as_of - dates).dt.days.clip(lower=0).to_numpy() / half_life_days)
    opponent_keys = np.where(is_home, matches["away_key"].to_numpy(), matches["home_key"].to_numpy())
    opponent_quality = np.array(
        [float((adversary_quality or {}).get(k, 0.0)) for k in opponent_keys],
        dtype=float,
    )
    quality_weights = weights * np.clip(1.0 + ADVERSARY_QUALITY_WEIGHT * opponent_quality, 0.55, 1.55)

    return {
        "matches": float(len(matches)),
        "ppg": float(np.average(pts, weights=weights)),
        "gf": float(np.average(gf, weights=weights)),
        "ga": float(np.average(ga, weights=weights)),
        "gd": float(np.average(gf - ga, weights=weights)),
        "win_rate": float(np.average(gf > ga, weights=weights)),
        "quality_ppg": float(np.average(pts, weights=quality_weights)),
        "quality_gd": float(np.average(gf - ga, weights=quality_weights)),
        "adversary_quality": float(np.average(opponent_quality, weights=weights)),
    }


def _fifa_tier_goal_summary(
    results: pd.DataFrame,
    key: str,
    world_cup: int,
    target_opponent_points: float,
) -> dict[str, float]:
    as_of = TOURNAMENT_STARTS[world_cup]
    before = results[results["date"] < as_of]
    matches = _team_slice(before, key).tail(FIFA_TIER_RECENT_N)
    if matches.empty:
        return {"gf": 1.25, "ga": 1.25, "gd": 0.0, "support": 0.0}

    point_map = _fifa_points_map(world_cup)
    point_median = float(np.median(list(point_map.values()))) if point_map else float(target_opponent_points)
    is_home = matches["home_key"].eq(key).to_numpy()
    opponent_keys = np.where(is_home, matches["away_key"].to_numpy(), matches["home_key"].to_numpy())
    opponent_points = np.array([float(point_map.get(k, point_median)) for k in opponent_keys], dtype=float)
    gf = np.where(is_home, matches["home_score"].to_numpy(), matches["away_score"].to_numpy())
    ga = np.where(is_home, matches["away_score"].to_numpy(), matches["home_score"].to_numpy())
    recency = 0.5 ** (
        (as_of - pd.to_datetime(matches["date"])).dt.days.clip(lower=0).to_numpy() / FIFA_TIER_HALFLIFE_DAYS
    )
    rating_similarity = 0.5 ** (np.abs(opponent_points - float(target_opponent_points)) / FIFA_TIER_POINTS_HALFLIFE)
    weights = recency * rating_similarity
    if float(weights.sum()) <= 1e-9:
        base = _summarize_team_matches(matches, key, as_of, RECENT_HALFLIFE_DAYS)
        return {"gf": base["gf"], "ga": base["ga"], "gd": base["gd"], "support": 0.0}

    return {
        "gf": float(np.average(gf, weights=weights)),
        "ga": float(np.average(ga, weights=weights)),
        "gd": float(np.average(gf - ga, weights=weights)),
        "support": float(rating_similarity.sum()),
    }


def _recent_adversary_quality(results: pd.DataFrame, world_cup: int) -> dict[str, float]:
    as_of = TOURNAMENT_STARTS[world_cup]
    cache_key = (world_cup, str(results["date"].max().date()), len(results))
    if cache_key in _QUALITY_CACHE:
        return _QUALITY_CACHE[cache_key]

    before = results[results["date"] < as_of]
    keys = sorted(set(before["home_key"]) | set(before["away_key"]))
    raw = {}
    for key in keys:
        matches = _team_slice(before, key).tail(RECENT_N)
        summary = _summarize_team_matches(matches, key, as_of, RECENT_HALFLIFE_DAYS)
        raw[key] = summary["ppg"] + 0.35 * summary["gd"]

    values = np.array(list(raw.values()), dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    std = float(values.std()) or 1.0
    out = {key: float(np.clip((value - mean) / std, -3.0, 3.0)) for key, value in raw.items()}
    _QUALITY_CACHE[cache_key] = out
    return out


def _summarize_elite_recent(
    matches: pd.DataFrame,
    key: str,
    as_of: pd.Timestamp,
    adversary_quality: dict[str, float],
) -> dict[str, float]:
    if matches.empty:
        return {
            "matches": 0.0,
            "ppg": ELITE_RESULT_PRIOR_PPG,
            "gd": ELITE_RESULT_PRIOR_GD,
        }

    is_home = matches["home_key"].eq(key).to_numpy()
    opponent_keys = np.where(is_home, matches["away_key"].to_numpy(), matches["home_key"].to_numpy())
    opponent_quality = np.array([float(adversary_quality.get(k, 0.0)) for k in opponent_keys], dtype=float)
    elite_mask = opponent_quality >= ELITE_ADVERSARY_THRESHOLD
    if not elite_mask.any():
        return {
            "matches": 0.0,
            "ppg": ELITE_RESULT_PRIOR_PPG,
            "gd": ELITE_RESULT_PRIOR_GD,
        }

    elite = matches.loc[elite_mask]
    is_home = elite["home_key"].eq(key).to_numpy()
    gf = np.where(is_home, elite["home_score"].to_numpy(), elite["away_score"].to_numpy())
    ga = np.where(is_home, elite["away_score"].to_numpy(), elite["home_score"].to_numpy())
    pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    dates = pd.to_datetime(elite["date"])
    elite_quality = opponent_quality[elite_mask]
    weights = 0.5 ** ((as_of - dates).dt.days.clip(lower=0).to_numpy() / RECENT_HALFLIFE_DAYS)
    weights = weights * np.clip(1.0 + ADVERSARY_QUALITY_WEIGHT * elite_quality, 0.75, 1.75)
    return {
        "matches": float(len(elite)),
        "ppg": float(np.average(pts, weights=weights)),
        "gd": float(np.average(gf - ga, weights=weights)),
    }


def team_features(
    results: pd.DataFrame,
    key: str,
    world_cup: int,
    recent_n: int = RECENT_N,
    short_recent_n: int = SHORT_RECENT_N,
    previous_wc_match_n: int = 16,
) -> dict[str, float]:
    as_of = TOURNAMENT_STARTS[world_cup]
    cycle_start = QUALIFYING_STARTS[world_cup]
    before = results[results["date"] < as_of]
    adversary_quality = _recent_adversary_quality(results, world_cup)

    recent = _team_slice(before, key).tail(recent_n)
    short_recent = _team_slice(before, key).tail(short_recent_n)
    cycle = _team_slice(before[before["date"] >= cycle_start], key)
    cycle = cycle[cycle["tournament"].astype(str).str.contains("FIFA World Cup qualification", case=False)]
    wc_hist = _team_slice(before, key)
    wc_hist = wc_hist[
        wc_hist["tournament"].astype(str).str.fullmatch("FIFA World Cup", case=False, na=False)
    ].tail(previous_wc_match_n)

    rec = _summarize_team_matches(recent, key, as_of, RECENT_HALFLIFE_DAYS, adversary_quality)
    short = _summarize_team_matches(short_recent, key, as_of, SHORT_RECENT_HALFLIFE_DAYS, adversary_quality)
    elite = _summarize_elite_recent(recent, key, as_of, adversary_quality)
    cyc = _summarize_team_matches(cycle, key, as_of, CYCLE_HALFLIFE_DAYS, adversary_quality)
    prev_wc = _summarize_team_matches(wc_hist, key, as_of, PREVIOUS_WC_HALFLIFE_DAYS)
    return {
        "recent_matches": rec["matches"],
        "recent_ppg": rec["ppg"],
        "recent_gf": rec["gf"],
        "recent_ga": rec["ga"],
        "recent_gd": rec["gd"],
        "recent_win_rate": rec["win_rate"],
        "short_ppg": short["ppg"],
        "short_gf": short["gf"],
        "short_ga": short["ga"],
        "short_gd": short["gd"],
        "recent_quality_ppg": rec["quality_ppg"],
        "recent_quality_gd": rec["quality_gd"],
        "recent_adversary_quality": rec["adversary_quality"],
        "elite_recent_matches": elite["matches"],
        "elite_recent_ppg": elite["ppg"],
        "elite_recent_gd": elite["gd"],
        "cycle_matches": cyc["matches"],
        "cycle_ppg": cyc["ppg"],
        "cycle_gd": cyc["gd"],
        "prev_wc_matches": prev_wc["matches"],
        "prev_wc_ppg": prev_wc["ppg"],
        "prev_wc_gd": prev_wc["gd"],
    }


def _perspective_features(
    row: pd.Series,
    own_prefix: str,
    opp_prefix: str,
    results: pd.DataFrame,
) -> dict[str, float]:
    own_key = row[f"{own_prefix}_key"]
    opp_key = row[f"{opp_prefix}_key"]
    own = team_features(results, own_key, int(row["world_cup"]))
    opp = team_features(results, opp_key, int(row["world_cup"]))
    host_set = HOSTS[int(row["world_cup"])]

    own_points = float(row[f"{own_prefix}_fifa_points"])
    opp_points = float(row[f"{opp_prefix}_fifa_points"])
    own_rank = float(row[f"{own_prefix}_fifa_rank"])
    opp_rank = float(row[f"{opp_prefix}_fifa_rank"])
    own_tier = _fifa_tier_goal_summary(results, own_key, int(row["world_cup"]), opp_points)
    opp_tier = _fifa_tier_goal_summary(results, opp_key, int(row["world_cup"]), own_points)

    return {
        "fifa_points_diff": own_points - opp_points,
        "fifa_rank_advantage": opp_rank - own_rank,
        "recent_ppg_diff": own["recent_ppg"] - opp["recent_ppg"],
        "recent_gd_diff": own["recent_gd"] - opp["recent_gd"],
        "short_ppg_diff": own["short_ppg"] - opp["short_ppg"],
        "short_gd_diff": own["short_gd"] - opp["short_gd"],
        "own_recent_ppg": own["recent_ppg"],
        "own_recent_gf": own["recent_gf"],
        "own_recent_ga": own["recent_ga"],
        "own_recent_gd": own["recent_gd"],
        "own_short_ppg": own["short_ppg"],
        "own_short_gf": own["short_gf"],
        "own_short_ga": own["short_ga"],
        "own_short_gd": own["short_gd"],
        "own_recent_quality_ppg": own["recent_quality_ppg"],
        "own_recent_quality_gd": own["recent_quality_gd"],
        "own_recent_adversary_quality": own["recent_adversary_quality"],
        "own_elite_recent_ppg": own["elite_recent_ppg"],
        "own_elite_recent_gd": own["elite_recent_gd"],
        "own_elite_recent_matches": own["elite_recent_matches"],
        "opp_recent_ppg": opp["recent_ppg"],
        "opp_recent_gf": opp["recent_gf"],
        "opp_recent_ga": opp["recent_ga"],
        "opp_recent_gd": opp["recent_gd"],
        "opp_short_ppg": opp["short_ppg"],
        "opp_short_gf": opp["short_gf"],
        "opp_short_ga": opp["short_ga"],
        "opp_short_gd": opp["short_gd"],
        "opp_recent_quality_ppg": opp["recent_quality_ppg"],
        "opp_recent_quality_gd": opp["recent_quality_gd"],
        "opp_recent_adversary_quality": opp["recent_adversary_quality"],
        "opp_elite_recent_ppg": opp["elite_recent_ppg"],
        "opp_elite_recent_gd": opp["elite_recent_gd"],
        "opp_elite_recent_matches": opp["elite_recent_matches"],
        "quality_recent_ppg_diff": own["recent_quality_ppg"] - opp["recent_quality_ppg"],
        "quality_recent_gd_diff": own["recent_quality_gd"] - opp["recent_quality_gd"],
        "adversary_quality_diff": own["recent_adversary_quality"] - opp["recent_adversary_quality"],
        "elite_recent_ppg_diff": own["elite_recent_ppg"] - opp["elite_recent_ppg"],
        "elite_recent_gd_diff": own["elite_recent_gd"] - opp["elite_recent_gd"],
        "own_fifa_tier_gf": own_tier["gf"],
        "own_fifa_tier_ga": own_tier["ga"],
        "own_fifa_tier_gd": own_tier["gd"],
        "own_fifa_tier_support": own_tier["support"],
        "opp_fifa_tier_gf": opp_tier["gf"],
        "opp_fifa_tier_ga": opp_tier["ga"],
        "opp_fifa_tier_gd": opp_tier["gd"],
        "opp_fifa_tier_support": opp_tier["support"],
        "fifa_tier_attack_edge": own_tier["gf"] - opp_tier["ga"],
        "fifa_tier_defense_edge": opp_tier["gf"] - own_tier["ga"],
        "own_cycle_ppg": own["cycle_ppg"],
        "own_cycle_gd": own["cycle_gd"],
        "opp_cycle_ppg": opp["cycle_ppg"],
        "opp_cycle_gd": opp["cycle_gd"],
        "own_prev_wc_ppg": own["prev_wc_ppg"],
        "own_prev_wc_gd": own["prev_wc_gd"],
        "opp_prev_wc_ppg": opp["prev_wc_ppg"],
        "opp_prev_wc_gd": opp["prev_wc_gd"],
        "host_diff": float(own_key in host_set) - float(opp_key in host_set),
        "is_knockout": float(row["is_knockout"]),
    }


def _training_weight(world_cup: int, is_knockout: int | float) -> float:
    recency_weight = WC_SAMPLE_RECENCY_WEIGHT.get(int(world_cup), 1.0)
    knockout_weight = 1.15 if float(is_knockout) else 1.0
    return float(recency_weight * knockout_weight)


def build_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = load_full_results()
    matches = load_worldcup_targets()
    match_rows = []
    goal_rows = []

    for row in matches.itertuples(index=False):
        r = pd.Series(row._asdict())
        home = _perspective_features(r, "home", "away", results)
        away = _perspective_features(r, "away", "home", results)
        outcome = "home_win" if r.home_score > r.away_score else "away_win" if r.away_score > r.home_score else "draw"

        match_row = {
            "date": r.date,
            "world_cup": int(r.world_cup),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_score": int(r.home_score),
            "away_score": int(r.away_score),
            "outcome": outcome,
            "sample_weight": _training_weight(int(r.world_cup), r.is_knockout),
            **{f"home_{k}": v for k, v in home.items()},
            **{f"away_{k}": v for k, v in away.items()},
        }
        for c in FEATURE_COLUMNS:
            match_row[c] = home[c]
        match_rows.append(match_row)

        goal_rows.append(
            {
                "date": r.date,
                "world_cup": int(r.world_cup),
                "team": r.home_team,
                "opponent": r.away_team,
                "goals": int(r.home_score),
                "side": "home",
                "sample_weight": _training_weight(int(r.world_cup), r.is_knockout),
                **home,
            }
        )
        goal_rows.append(
            {
                "date": r.date,
                "world_cup": int(r.world_cup),
                "team": r.away_team,
                "opponent": r.home_team,
                "goals": int(r.away_score),
                "side": "away",
                "sample_weight": _training_weight(int(r.world_cup), r.is_knockout),
                **away,
            }
        )

    return pd.DataFrame(match_rows), pd.DataFrame(goal_rows)


def fit_models(match_features: pd.DataFrame, goal_rows: pd.DataFrame) -> LearnedModels:
    goal_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.8, max_iter=2000)),
        ]
    )
    outcome_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.7, max_iter=2000)),
        ]
    )
    goal_sample_weight = goal_rows["sample_weight"] if "sample_weight" in goal_rows else None
    outcome_sample_weight = match_features["sample_weight"] if "sample_weight" in match_features else None
    goal_model.fit(
        goal_rows[FEATURE_COLUMNS],
        goal_rows["goals"],
        model__sample_weight=goal_sample_weight,
    )
    outcome_model.fit(
        match_features[FEATURE_COLUMNS],
        match_features["outcome"],
        model__sample_weight=outcome_sample_weight,
    )
    return LearnedModels(goal_model, outcome_model, FEATURE_COLUMNS)


def poisson_matrix(mu_home: float, mu_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    home = np.exp(-mu_home) * np.power(mu_home, goals) / np.array([math.factorial(int(g)) for g in goals])
    away = np.exp(-mu_away) * np.power(mu_away, goals) / np.array([math.factorial(int(g)) for g in goals])
    mat = np.outer(home, away)
    return mat / mat.sum()


def poisson_outcome_probs(mu_home: float, mu_away: float) -> tuple[float, float, float]:
    mat = poisson_matrix(mu_home, mu_away)
    n = mat.shape[0]
    home_win = mat[np.tril_indices(n, -1)].sum()
    away_win = mat[np.triu_indices(n, 1)].sum()
    draw = np.trace(mat)
    return float(home_win), float(draw), float(away_win)


def modal_score(mu_home: float, mu_away: float) -> tuple[int, int]:
    mat = poisson_matrix(mu_home, mu_away)
    return tuple(int(x) for x in np.unravel_index(np.argmax(mat), mat.shape))


def best_group_score(mu_home: float, mu_away: float) -> tuple[int, int]:
    mat = poisson_matrix(mu_home, mu_away)
    n = mat.shape[0]
    ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best, best_ev = (1, 1), -1.0
    for pi in range(n):
        for pj in range(n):
            exact = mat[pi, pj] * 25
            gd = ((pi - pj) == (ai - aj)) & ~((ai == pi) & (aj == pj))
            total = ((pi + pj) == (ai + aj)) & ~((ai == pi) & (aj == pj))
            ev = exact + 10 * (mat[gd].sum() + mat[total].sum())
            pred_outcome = 0 if pi > pj else 1 if pi < pj else 2
            actual_outcome = np.where(ai > aj, 0, np.where(ai < aj, 1, 2))
            ev += 40 * mat[actual_outcome == pred_outcome].sum()
            if ev > best_ev:
                best_ev = ev
                best = (pi, pj)
    return best


def best_knockout_score(mu_home: float, mu_away: float) -> tuple[int, int]:
    mat = poisson_matrix(mu_home, mu_away)
    p_home, p_draw, p_away = poisson_outcome_probs(mu_home, mu_away)
    draw_pick = abs(p_home - p_away) < KNOCKOUT_DRAW_EDGE_THRESHOLD
    fav_home = p_home >= p_away
    n = mat.shape[0]
    ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best, best_ev = ((1, 1) if draw_pick else (1, 0)), -1.0
    for pi in range(n):
        for pj in range(n):
            if draw_pick and pi != pj:
                continue
            if not draw_pick:
                if fav_home and pi <= pj:
                    continue
                if not fav_home and pj <= pi:
                    continue
            exact = mat[pi, pj] * 25
            gd = ((pi - pj) == (ai - aj)) & ~((ai == pi) & (aj == pj))
            total = ((pi + pj) == (ai + aj)) & ~((ai == pi) & (aj == pj))
            ev = exact + 10 * (mat[gd].sum() + mat[total].sum())
            if ev > best_ev:
                best_ev = ev
                best = (pi, pj)
    return best


def predict_matches(models: LearnedModels, match_features: pd.DataFrame) -> pd.DataFrame:
    home_features = match_features[FEATURE_COLUMNS].copy()
    away_feature_rows = []
    for _, r in match_features.iterrows():
        away_feature_rows.append({c: r[f"away_{c}"] for c in FEATURE_COLUMNS})
    away_features = pd.DataFrame(away_feature_rows)

    mu_home = models.goal_model.predict(home_features)
    mu_away = models.goal_model.predict(away_features)
    class_order = list(models.outcome_model.named_steps["model"].classes_)
    class_probs = models.outcome_model.predict_proba(home_features)

    rows = []
    has_actual = {"home_score", "away_score", "outcome"}.issubset(match_features.columns)
    for i, r in match_features.reset_index(drop=True).iterrows():
        ph, pd_, pa = poisson_outcome_probs(float(mu_home[i]), float(mu_away[i]))
        pg = {"home_win": ph, "draw": pd_, "away_win": pa}
        pred_outcome = max(pg, key=pg.get)
        score_h, score_a = modal_score(float(mu_home[i]), float(mu_away[i]))
        logit_probs = {f"logit_p_{cls}": float(class_probs[i, j]) for j, cls in enumerate(class_order)}
        out = {
            "date": r["date"],
            "world_cup": int(r["world_cup"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "pred_home_goals_mean": float(mu_home[i]),
            "pred_away_goals_mean": float(mu_away[i]),
            "pred_home_goals_modal": score_h,
            "pred_away_goals_modal": score_a,
            "pred_outcome_poisson": pred_outcome,
            "poisson_p_home_win": ph,
            "poisson_p_draw": pd_,
            "poisson_p_away_win": pa,
            "pred_outcome_logit": class_order[int(np.argmax(class_probs[i]))],
            **logit_probs,
        }
        if has_actual:
            out.update(
                {
                    "home_score": int(r["home_score"]),
                    "away_score": int(r["away_score"]),
                    "actual_outcome": r["outcome"],
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def score_predictions(pred: pd.DataFrame) -> dict[str, float]:
    actual_goals = np.r_[pred["home_score"].to_numpy(), pred["away_score"].to_numpy()]
    pred_goals = np.r_[pred["pred_home_goals_mean"].to_numpy(), pred["pred_away_goals_mean"].to_numpy()]
    exact = (
        (pred["home_score"] == pred["pred_home_goals_modal"])
        & (pred["away_score"] == pred["pred_away_goals_modal"])
    ).mean()
    poisson_acc = accuracy_score(pred["actual_outcome"], pred["pred_outcome_poisson"])
    logit_acc = accuracy_score(pred["actual_outcome"], pred["pred_outcome_logit"])

    logit_cols = sorted([c for c in pred.columns if c.startswith("logit_p_")])
    labels = [c.replace("logit_p_", "") for c in logit_cols]
    return {
        "matches": int(len(pred)),
        "goal_mae": float(mean_absolute_error(actual_goals, pred_goals)),
        "modal_exact_score_rate": float(exact),
        "poisson_outcome_accuracy": float(poisson_acc),
        "logit_outcome_accuracy": float(logit_acc),
        "logit_log_loss": float(log_loss(pred["actual_outcome"], pred[logit_cols], labels=labels)),
    }


def temporal_backtest(match_features: pd.DataFrame, goal_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    preds = []
    folds = []
    for test_wc in [2018, 2022]:
        train_matches = match_features[match_features["world_cup"] < test_wc]
        train_goals = goal_rows[goal_rows["world_cup"] < test_wc]
        test_matches = match_features[match_features["world_cup"] == test_wc]
        models = fit_models(train_matches, train_goals)
        pred = predict_matches(models, test_matches)
        pred["backtest_type"] = "temporal"
        pred["train_world_cups"] = ",".join(str(v) for v in sorted(train_matches["world_cup"].unique()))
        preds.append(pred)
        fold_metrics = score_predictions(pred)
        fold_metrics["test_world_cup"] = int(test_wc)
        folds.append(fold_metrics)

    out = pd.concat(preds, ignore_index=True)
    return out, {"folds": folds, "overall": score_predictions(out)}


def leave_one_worldcup_out(match_features: pd.DataFrame, goal_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    preds = []
    folds = []
    for test_wc in sorted(match_features["world_cup"].unique()):
        train_matches = match_features[match_features["world_cup"] != test_wc]
        train_goals = goal_rows[goal_rows["world_cup"] != test_wc]
        test_matches = match_features[match_features["world_cup"] == test_wc]
        models = fit_models(train_matches, train_goals)
        pred = predict_matches(models, test_matches)
        pred["backtest_type"] = "leave_one_worldcup_out"
        pred["train_world_cups"] = ",".join(str(v) for v in sorted(train_matches["world_cup"].unique()))
        preds.append(pred)
        fold_metrics = score_predictions(pred)
        fold_metrics["test_world_cup"] = int(test_wc)
        folds.append(fold_metrics)

    out = pd.concat(preds, ignore_index=True)
    return out, {"folds": folds, "overall": score_predictions(out)}


def coefficient_tables(models: LearnedModels) -> tuple[pd.DataFrame, pd.DataFrame]:
    goal_reg = models.goal_model.named_steps["model"]
    goal_coef = pd.DataFrame(
        {
            "model": "poisson_goals",
            "target": "own_goals",
            "feature": FEATURE_COLUMNS,
            "coefficient": goal_reg.coef_,
        }
    )
    goal_coef["abs_coefficient"] = goal_coef["coefficient"].abs()
    goal_coef = goal_coef.sort_values("abs_coefficient", ascending=False)

    logit = models.outcome_model.named_steps["model"]
    rows = []
    for class_name, coefs in zip(logit.classes_, logit.coef_):
        for feature, coef in zip(FEATURE_COLUMNS, coefs):
            rows.append(
                {
                    "model": "logistic_outcome",
                    "target": class_name,
                    "feature": feature,
                    "coefficient": float(coef),
                    "abs_coefficient": abs(float(coef)),
                }
            )
    outcome_coef = pd.DataFrame(rows).sort_values(["target", "abs_coefficient"], ascending=[True, False])
    return goal_coef, outcome_coef


def train_and_export() -> dict:
    HIST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    match_features, goal_rows = build_feature_tables()
    match_features.to_csv(HIST / "learned_match_features.csv", index=False)
    goal_rows.to_csv(HIST / "learned_goal_training_rows.csv", index=False)

    temporal_pred, temporal_metrics = temporal_backtest(match_features, goal_rows)
    loo_pred, loo_metrics = leave_one_worldcup_out(match_features, goal_rows)
    pd.concat([temporal_pred, loo_pred], ignore_index=True).to_csv(
        HIST / "learned_model_backtest_predictions.csv", index=False
    )

    final_models = fit_models(match_features, goal_rows)
    final_pred = predict_matches(final_models, match_features)
    final_pred.to_csv(HIST / "learned_model_in_sample_predictions.csv", index=False)

    goal_coef, outcome_coef = coefficient_tables(final_models)
    goal_coef.to_csv(HIST / "learned_goal_weight_coefficients.csv", index=False)
    outcome_coef.to_csv(HIST / "learned_outcome_weight_coefficients.csv", index=False)

    with (OUT / "learned_worldcup_models.pkl").open("wb") as f:
        pickle.dump(final_models, f)

    metrics = {
        "feature_columns": FEATURE_COLUMNS,
        "recent_feature_settings": {
            "recent_n": RECENT_N,
            "short_recent_n": SHORT_RECENT_N,
            "recent_half_life_days": RECENT_HALFLIFE_DAYS,
            "short_recent_half_life_days": SHORT_RECENT_HALFLIFE_DAYS,
            "adversary_quality_weight": ADVERSARY_QUALITY_WEIGHT,
            "elite_adversary_threshold": ELITE_ADVERSARY_THRESHOLD,
            "fifa_tier_recent_n": FIFA_TIER_RECENT_N,
            "fifa_tier_half_life_days": FIFA_TIER_HALFLIFE_DAYS,
            "fifa_tier_points_half_life": FIFA_TIER_POINTS_HALFLIFE,
        },
        "submission_blend": {
            "learned_goals_weight": LEARNED_GOALS_WEIGHT,
            "previous_engine_goals_weight": PREVIOUS_ENGINE_GOALS_WEIGHT,
            "fifa_tier_direct_goals_weight": FIFA_TIER_DIRECT_GOALS_WEIGHT,
            "tournament_context_weight": TOURNAMENT_CONTEXT_WEIGHT,
            "current_form_context_weight": CURRENT_FORM_CONTEXT_WEIGHT,
            "current_form_context_clip": CURRENT_FORM_CONTEXT_CLIP,
            "pedigree_anchor_weight": PEDIGREE_ANCHOR_WEIGHT,
            "pedigree_edge_scale": PEDIGREE_EDGE_SCALE,
            "mismatch_edge_scale": MISMATCH_EDGE_SCALE,
            "mismatch_total_goal_lift": MISMATCH_TOTAL_GOAL_LIFT,
            "mismatch_favorite_goal_lift": MISMATCH_FAVORITE_GOAL_LIFT,
            "recent_tie_break_edge": RECENT_TIE_BREAK_EDGE,
            "knockout_draw_edge_threshold": KNOCKOUT_DRAW_EDGE_THRESHOLD,
        },
        "world_cup_sample_recency_weight": WC_SAMPLE_RECENCY_WEIGHT,
        "training_world_cups": sorted(int(v) for v in match_features["world_cup"].unique()),
        "training_matches": int(len(match_features)),
        "goal_training_rows": int(len(goal_rows)),
        "temporal_backtest": temporal_metrics,
        "leave_one_worldcup_out": loo_metrics,
        "in_sample": score_predictions(final_pred),
    }
    (HIST / "learned_model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_exported_models() -> LearnedModels:
    model_path = OUT / "learned_worldcup_models.pkl"
    if not model_path.exists():
        train_and_export()
    with model_path.open("rb") as f:
        models = pickle.load(f)
    if getattr(models, "feature_columns", None) != FEATURE_COLUMNS:
        train_and_export()
        with model_path.open("rb") as f:
            models = pickle.load(f)
    return models


def build_2026_group_feature_table() -> pd.DataFrame:
    results = load_full_results()
    fixtures = pd.read_csv(GROUP_FIXTURES)
    date_col = "date_utc" if "date_utc" in fixtures.columns else "date"
    fixtures[date_col] = pd.to_datetime(fixtures[date_col], errors="coerce")
    ranking = pd.read_csv(CURRENT_RANKING).set_index("team_key")

    rows = []
    for r in fixtures.itertuples(index=False):
        home_key = team_key(r.home_team)
        away_key = team_key(r.away_team)
        home_rank = ranking.loc[home_key]
        away_rank = ranking.loc[away_key]
        fixture_date = getattr(r, date_col)
        base = pd.Series(
            {
                "date": fixture_date,
                "world_cup": 2026,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_key": home_key,
                "away_key": away_key,
                "home_fifa_rank": float(home_rank["rank"]),
                "home_fifa_points": float(home_rank["points"]),
                "away_fifa_rank": float(away_rank["rank"]),
                "away_fifa_points": float(away_rank["points"]),
                "is_knockout": 0,
            }
        )
        home = _perspective_features(base, "home", "away", results)
        away = _perspective_features(base, "away", "home", results)
        row = {
            "match_id": r.match_id,
            "group": r.group,
            "date": fixture_date,
            "venue": r.venue,
            "world_cup": 2026,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_key": home_key,
            "away_key": away_key,
            **{f"home_{k}": v for k, v in home.items()},
            **{f"away_{k}": v for k, v in away.items()},
        }
        for c in FEATURE_COLUMNS:
            row[c] = home[c]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_2026_group_stage() -> pd.DataFrame:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    models = load_exported_models()
    features = build_2026_group_feature_table()
    features.to_csv(HIST / "learned_2026_group_features.csv", index=False)
    pred = predict_matches(models, features)
    out = features[["match_id", "group", "date", "home_team", "away_team", "venue"]].copy()
    out["predicted_home_goals"] = pred["pred_home_goals_modal"]
    out["predicted_away_goals"] = pred["pred_away_goals_modal"]
    out["predicted_home_goals_mean"] = pred["pred_home_goals_mean"]
    out["predicted_away_goals_mean"] = pred["pred_away_goals_mean"]
    out["winning_team"] = pred["pred_outcome_poisson"].map(
        {"home_win": "home", "away_win": "away", "draw": "draw"}
    )
    out["home_win_probability"] = pred["poisson_p_home_win"]
    out["draw_probability"] = pred["poisson_p_draw"]
    out["away_win_probability"] = pred["poisson_p_away_win"]
    out.to_csv(PREDICTIONS / "group_predictions_learned_weights.csv", index=False)
    return out


class LearnedSubmissionEngine:
    def __init__(self, n_sims: int = 20000, verbose: bool = True):
        self.n_sims = n_sims
        self.verbose = verbose
        self.models = load_exported_models()
        self.previous_engine = PreviousEngine(n_sims=1, verbose=False)
        self.results = load_full_results()
        self.ranking = pd.read_csv(CURRENT_RANKING).set_index("team_key")
        self.gf = pd.read_csv(GROUP_FIXTURES)
        self.ks = pd.read_csv(KNOCKOUT_SLOTS)
        self.gf["date_utc"] = pd.to_datetime(self.gf["date_utc"], errors="coerce")
        self.ks["date_utc"] = pd.to_datetime(self.ks["date_utc"], errors="coerce")

        cells = sorted(set(self.gf["home_team"]) | set(self.gf["away_team"]))
        self.cell_key = {cell: team_key(cell) for cell in cells}
        self.keys = sorted(set(self.cell_key.values()))
        self.kidx = {key: i for i, key in enumerate(self.keys)}
        self.disp = {
            key: str(self.ranking.loc[key, "team"]) if key in self.ranking.index else key.title()
            for key in self.keys
        }
        self.points_by_key = {
            key: float(self.ranking.loc[key, "points"]) if key in self.ranking.index else float(self.ranking["points"].median())
            for key in self.keys
        }
        self.rank_by_key = {
            key: float(self.ranking.loc[key, "rank"]) if key in self.ranking.index else float(self.ranking["rank"].median())
            for key in self.keys
        }
        self.team_feature_cache = {
            key: team_features(self.results, key, 2026)
            for key in self.keys
        }
        self.fifa_tier_feature_cache: dict[tuple[str, float], dict[str, float]] = {}
        self._precompute_lambdas()

    def _fifa_tier_summary(self, own_key: str, target_opponent_points: float) -> dict[str, float]:
        cache_key = (own_key, round(float(target_opponent_points), 2))
        if cache_key not in self.fifa_tier_feature_cache:
            self.fifa_tier_feature_cache[cache_key] = _fifa_tier_goal_summary(
                self.results,
                own_key,
                2026,
                target_opponent_points,
            )
        return self.fifa_tier_feature_cache[cache_key]

    def _feature_row(self, own_key: str, opp_key: str, is_knockout: bool) -> dict[str, float]:
        own = self.team_feature_cache[own_key]
        opp = self.team_feature_cache[opp_key]
        own_tier = self._fifa_tier_summary(own_key, self.points_by_key[opp_key])
        opp_tier = self._fifa_tier_summary(opp_key, self.points_by_key[own_key])
        host_set = HOSTS[2026]
        return {
            "fifa_points_diff": self.points_by_key[own_key] - self.points_by_key[opp_key],
            "fifa_rank_advantage": self.rank_by_key[opp_key] - self.rank_by_key[own_key],
            "recent_ppg_diff": own["recent_ppg"] - opp["recent_ppg"],
            "recent_gd_diff": own["recent_gd"] - opp["recent_gd"],
            "short_ppg_diff": own["short_ppg"] - opp["short_ppg"],
            "short_gd_diff": own["short_gd"] - opp["short_gd"],
            "own_recent_ppg": own["recent_ppg"],
            "own_recent_gf": own["recent_gf"],
            "own_recent_ga": own["recent_ga"],
            "own_recent_gd": own["recent_gd"],
            "own_short_ppg": own["short_ppg"],
            "own_short_gf": own["short_gf"],
            "own_short_ga": own["short_ga"],
            "own_short_gd": own["short_gd"],
        "own_recent_quality_ppg": own["recent_quality_ppg"],
        "own_recent_quality_gd": own["recent_quality_gd"],
        "own_recent_adversary_quality": own["recent_adversary_quality"],
        "own_elite_recent_ppg": own["elite_recent_ppg"],
        "own_elite_recent_gd": own["elite_recent_gd"],
        "own_elite_recent_matches": own["elite_recent_matches"],
        "opp_recent_ppg": opp["recent_ppg"],
        "opp_recent_gf": opp["recent_gf"],
        "opp_recent_ga": opp["recent_ga"],
            "opp_recent_gd": opp["recent_gd"],
            "opp_short_ppg": opp["short_ppg"],
            "opp_short_gf": opp["short_gf"],
            "opp_short_ga": opp["short_ga"],
            "opp_short_gd": opp["short_gd"],
        "opp_recent_quality_ppg": opp["recent_quality_ppg"],
        "opp_recent_quality_gd": opp["recent_quality_gd"],
        "opp_recent_adversary_quality": opp["recent_adversary_quality"],
        "opp_elite_recent_ppg": opp["elite_recent_ppg"],
        "opp_elite_recent_gd": opp["elite_recent_gd"],
        "opp_elite_recent_matches": opp["elite_recent_matches"],
        "quality_recent_ppg_diff": own["recent_quality_ppg"] - opp["recent_quality_ppg"],
        "quality_recent_gd_diff": own["recent_quality_gd"] - opp["recent_quality_gd"],
        "adversary_quality_diff": own["recent_adversary_quality"] - opp["recent_adversary_quality"],
            "elite_recent_ppg_diff": own["elite_recent_ppg"] - opp["elite_recent_ppg"],
            "elite_recent_gd_diff": own["elite_recent_gd"] - opp["elite_recent_gd"],
            "own_fifa_tier_gf": own_tier["gf"],
            "own_fifa_tier_ga": own_tier["ga"],
            "own_fifa_tier_gd": own_tier["gd"],
            "own_fifa_tier_support": own_tier["support"],
            "opp_fifa_tier_gf": opp_tier["gf"],
            "opp_fifa_tier_ga": opp_tier["ga"],
            "opp_fifa_tier_gd": opp_tier["gd"],
            "opp_fifa_tier_support": opp_tier["support"],
            "fifa_tier_attack_edge": own_tier["gf"] - opp_tier["ga"],
            "fifa_tier_defense_edge": opp_tier["gf"] - own_tier["ga"],
            "own_cycle_ppg": own["cycle_ppg"],
        "own_cycle_gd": own["cycle_gd"],
            "opp_cycle_ppg": opp["cycle_ppg"],
            "opp_cycle_gd": opp["cycle_gd"],
            "own_prev_wc_ppg": own["prev_wc_ppg"],
            "own_prev_wc_gd": own["prev_wc_gd"],
            "opp_prev_wc_ppg": opp["prev_wc_ppg"],
            "opp_prev_wc_gd": opp["prev_wc_gd"],
            "host_diff": float(own_key in host_set) - float(opp_key in host_set),
            "is_knockout": float(is_knockout),
        }

    def _matchup_edge(self, home_key: str, away_key: str) -> float:
        home = self.team_feature_cache[home_key]
        away = self.team_feature_cache[away_key]
        home_tier = self._fifa_tier_summary(home_key, self.points_by_key[away_key])
        away_tier = self._fifa_tier_summary(away_key, self.points_by_key[home_key])
        tier_edge = (home_tier["gf"] - away_tier["ga"]) - (away_tier["gf"] - home_tier["ga"])
        fifa_edge = (self.points_by_key[home_key] - self.points_by_key[away_key]) / 120.0
        recent_edge = (
            0.45 * (home["short_gd"] - away["short_gd"])
            + 0.35 * (home["elite_recent_gd"] - away["elite_recent_gd"])
            + 0.25 * (home["recent_quality_gd"] - away["recent_quality_gd"])
            + 0.20 * (home["short_ppg"] - away["short_ppg"])
            + 0.10 * (home["recent_adversary_quality"] - away["recent_adversary_quality"])
            + 0.35 * tier_edge
        )
        return float(np.clip(0.45 * fifa_edge + 0.55 * recent_edge, -MISMATCH_EDGE_SCALE, MISMATCH_EDGE_SCALE))

    @staticmethod
    def _expand_mismatch_goals(mu_home: float, mu_away: float, edge: float) -> tuple[float, float]:
        mismatch = abs(edge) / MISMATCH_EDGE_SCALE
        if mismatch <= 0:
            return mu_home, mu_away

        total_lift = 1.0 + MISMATCH_TOTAL_GOAL_LIFT * mismatch
        favorite_lift = 1.0 + MISMATCH_FAVORITE_GOAL_LIFT * mismatch
        underdog_lift = 1.0 - MISMATCH_UNDERDOG_GOAL_TRIM * mismatch
        if edge >= 0:
            mu_home *= favorite_lift
            mu_away *= underdog_lift
        else:
            mu_home *= underdog_lift
            mu_away *= favorite_lift
        return mu_home * total_lift, mu_away * total_lift

    def _pedigree_edge(self, home_key: str, away_key: str) -> float:
        pedigree = self.previous_engine.pedigree_by_key
        median = self.previous_engine.rating_median
        return float((pedigree.get(home_key, median) - pedigree.get(away_key, median)) / PEDIGREE_EDGE_SCALE)

    def _direct_fifa_tier_goals(self, home_key: str, away_key: str) -> tuple[float, float]:
        home_tier = self._fifa_tier_summary(home_key, self.points_by_key[away_key])
        away_tier = self._fifa_tier_summary(away_key, self.points_by_key[home_key])
        home_goals = 0.60 * home_tier["gf"] + 0.40 * away_tier["ga"]
        away_goals = 0.60 * away_tier["gf"] + 0.40 * home_tier["ga"]
        return float(np.clip(home_goals, 0.15, 5.5)), float(np.clip(away_goals, 0.15, 5.5))

    def _apply_tournament_context(
        self,
        mu_home: float,
        mu_away: float,
        home_key: str,
        away_key: str,
    ) -> tuple[float, float]:
        home = self.team_feature_cache[home_key]
        away = self.team_feature_cache[away_key]
        tournament_edge = (
            0.55 * (home["prev_wc_gd"] - away["prev_wc_gd"])
            + 0.45 * (home["prev_wc_ppg"] - away["prev_wc_ppg"])
        )
        adjustment = float(np.clip(TOURNAMENT_CONTEXT_WEIGHT * tournament_edge, -0.14, 0.14))
        return mu_home * math.exp(adjustment), mu_away * math.exp(-adjustment)

    def _apply_current_form_context(
        self,
        mu_home: float,
        mu_away: float,
        home_key: str,
        away_key: str,
    ) -> tuple[float, float]:
        adjustment = float(np.clip(
            CURRENT_FORM_CONTEXT_WEIGHT * self._matchup_edge(home_key, away_key),
            -CURRENT_FORM_CONTEXT_CLIP,
            CURRENT_FORM_CONTEXT_CLIP,
        ))
        return mu_home * math.exp(adjustment), mu_away * math.exp(-adjustment)

    def _anchor_pedigree_goals(
        self,
        mu_home: float,
        mu_away: float,
        home_key: str,
        away_key: str,
    ) -> tuple[float, float]:
        total = max(mu_home + mu_away, 0.1)
        model_edge = math.log(max(mu_home, 1e-6) / max(mu_away, 1e-6))
        anchored_edge = (
            (1.0 - PEDIGREE_ANCHOR_WEIGHT) * model_edge
            + PEDIGREE_ANCHOR_WEIGHT * self._pedigree_edge(home_key, away_key)
        )
        ratio = math.exp(float(np.clip(anchored_edge, -2.4, 2.4)))
        anchored_home = total * ratio / (1.0 + ratio)
        anchored_away = total / (1.0 + ratio)
        return anchored_home, anchored_away

    def _ko_tie_winner(self, home_key: str, away_key: str, p_home: float, p_away: float) -> str:
        if abs(p_home - p_away) < KNOCKOUT_DRAW_EDGE_THRESHOLD:
            recent_edge = self._matchup_edge(home_key, away_key)
            if abs(recent_edge) >= RECENT_TIE_BREAK_EDGE:
                return "home" if recent_edge >= 0 else "away"
            return "home" if self._pedigree_edge(home_key, away_key) >= 0 else "away"
        return scoring.ko_winner(p_home, p_away)

    def _ko_tie_home_probability(self, home_key: str, away_key: str, p_home: float, p_away: float) -> float:
        if abs(p_home - p_away) < KNOCKOUT_DRAW_EDGE_THRESHOLD:
            recent_edge = self._matchup_edge(home_key, away_key)
            if abs(recent_edge) >= RECENT_TIE_BREAK_EDGE:
                return 1.0 if recent_edge >= 0 else 0.0
            return 1.0 if self._pedigree_edge(home_key, away_key) >= 0 else 0.0
        return p_home / max(p_home + p_away, 1e-9)

    def lambdas(self, home_key: str, away_key: str, is_knockout: bool = False) -> tuple[float, float]:
        home = pd.DataFrame([self._feature_row(home_key, away_key, is_knockout)])[FEATURE_COLUMNS]
        away = pd.DataFrame([self._feature_row(away_key, home_key, is_knockout)])[FEATURE_COLUMNS]
        learned_home = float(self.models.goal_model.predict(home)[0])
        learned_away = float(self.models.goal_model.predict(away)[0])
        previous_home, previous_away = self.previous_engine.lambdas(home_key, away_key, 0)
        tier_home, tier_away = self._direct_fifa_tier_goals(home_key, away_key)
        mu_home = (
            LEARNED_GOALS_WEIGHT * learned_home
            + PREVIOUS_ENGINE_GOALS_WEIGHT * previous_home
            + FIFA_TIER_DIRECT_GOALS_WEIGHT * tier_home
        )
        mu_away = (
            LEARNED_GOALS_WEIGHT * learned_away
            + PREVIOUS_ENGINE_GOALS_WEIGHT * previous_away
            + FIFA_TIER_DIRECT_GOALS_WEIGHT * tier_away
        )
        mu_home, mu_away = self._apply_tournament_context(mu_home, mu_away, home_key, away_key)
        mu_home, mu_away = self._anchor_pedigree_goals(mu_home, mu_away, home_key, away_key)
        mu_home, mu_away = self._apply_current_form_context(mu_home, mu_away, home_key, away_key)
        mu_home, mu_away = self._expand_mismatch_goals(mu_home, mu_away, self._matchup_edge(home_key, away_key))
        return float(np.clip(mu_home, 0.05, 6.0)), float(np.clip(mu_away, 0.05, 6.0))

    def _precompute_lambdas(self) -> None:
        n = len(self.keys)
        self.LH_GROUP = np.zeros((n, n))
        self.LA_GROUP = np.zeros((n, n))
        self.LH_KO = np.zeros((n, n))
        self.LA_KO = np.zeros((n, n))
        self.PW_KO = np.full((n, n), 0.5)
        for i, home_key in enumerate(self.keys):
            for j, away_key in enumerate(self.keys):
                if i == j:
                    continue
                self.LH_GROUP[i, j], self.LA_GROUP[i, j] = self.lambdas(home_key, away_key, False)
                self.LH_KO[i, j], self.LA_KO[i, j] = self.lambdas(home_key, away_key, True)
                p_home, _, p_away = poisson_outcome_probs(self.LH_KO[i, j], self.LA_KO[i, j])
                self.PW_KO[i, j] = self._ko_tie_home_probability(home_key, away_key, p_home, p_away)

    def match_extras(self, match_id: int, home_key: str, away_key: str, ko: bool) -> tuple[int, int, int]:
        rng = np.random.default_rng([SEED, int(match_id), 7919 if ko else 0])
        points = np.array([self.points_by_key[k] for k in self.keys])
        mu = float(points.mean())
        sd = float(points.std() or 1.0)
        home_z = (self.points_by_key[home_key] - mu) / sd
        away_z = (self.points_by_key[away_key] - mu) / sd
        corners = (
            CORNER_PER_TEAM_BASE + CORNER_RATING_COEF * home_z
            + CORNER_PER_TEAM_BASE + CORNER_RATING_COEF * away_z
            + rng.normal(0, 1.1)
        )
        yellows = (
            YELLOW_PER_TEAM_BASE - YELLOW_RATING_COEF * home_z
            + YELLOW_PER_TEAM_BASE - YELLOW_RATING_COEF * away_z
            + (0.7 if ko else 0.0)
            + rng.normal(0, 0.9)
        )
        red_p = RED_BASE + (RED_KO_BONUS if ko else 0.0)
        rr = rng.random()
        reds = (1 if rr < red_p else 0) + (1 if rr < red_p * 0.12 else 0)
        return int(round(np.clip(corners, 3, 17))), int(round(np.clip(yellows, 1, 9))), int(reds)

    @staticmethod
    def _bipartite(groups: list[str], slots_allowed: dict[str, set[str]]) -> dict[str, str | None]:
        slot_to_group = {slot: None for slot in slots_allowed}

        def assign(group: str, seen: set[str]) -> bool:
            for slot, allowed in slots_allowed.items():
                if group in allowed and slot not in seen:
                    seen.add(slot)
                    if slot_to_group[slot] is None or assign(slot_to_group[slot], seen):
                        slot_to_group[slot] = group
                        return True
            return False

        for group in groups:
            assign(group, set())
        return slot_to_group

    def simulate(self) -> dict[int, dict[str, float | int]]:
        rng = np.random.default_rng(SEED)
        n_sims = self.n_sims
        groups = sorted(self.gf["group"].unique())
        group_teams = {
            group: sorted({
                self.kidx[self.cell_key[cell]]
                for cell in set(self.gf[self.gf["group"] == group]["home_team"])
                | set(self.gf[self.gf["group"] == group]["away_team"])
            })
            for group in groups
        }
        stats = {group: np.zeros((len(group_teams[group]), 3, n_sims)) for group in groups}
        for group in groups:
            local_idx = {team_idx: i for i, team_idx in enumerate(group_teams[group])}
            for row in self.gf[self.gf["group"] == group].itertuples(index=False):
                home_idx = self.kidx[self.cell_key[row.home_team]]
                away_idx = self.kidx[self.cell_key[row.away_team]]
                mu_home = self.LH_GROUP[home_idx, away_idx]
                mu_away = self.LA_GROUP[home_idx, away_idx]
                home_goals = rng.poisson(mu_home, n_sims)
                away_goals = rng.poisson(mu_away, n_sims)
                home_pts = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
                away_pts = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))
                hloc = local_idx[home_idx]
                aloc = local_idx[away_idx]
                stats[group][hloc, 0] += home_pts
                stats[group][aloc, 0] += away_pts
                stats[group][hloc, 1] += home_goals - away_goals
                stats[group][aloc, 1] += away_goals - home_goals
                stats[group][hloc, 2] += home_goals
                stats[group][aloc, 2] += away_goals

        winners, runners_up, thirds, third_stats = {}, {}, {}, {}
        for group in groups:
            st = stats[group]
            arr = np.array(group_teams[group])
            score = st[:, 0] * 1e6 + st[:, 1] * 1e3 + st[:, 2] + rng.random((len(arr), n_sims)) * 1e-3
            order = np.argsort(-score, axis=0)
            winners[group], runners_up[group], thirds[group] = arr[order[0]], arr[order[1]], arr[order[2]]
            third_rows = order[2]
            third_stats[group] = np.stack([
                st[third_rows, 0, np.arange(n_sims)],
                st[third_rows, 1, np.arange(n_sims)],
                st[third_rows, 2, np.arange(n_sims)],
            ])

        third_score = np.stack([
            third_stats[group][0] * 1e6 + third_stats[group][1] * 1e3 + third_stats[group][2] + rng.random(n_sims) * 1e-3
            for group in groups
        ])
        best8 = np.argsort(-third_score, axis=0)[:8]
        best_third_slots = {}
        for slot in set(self.ks["slot_home"]) | set(self.ks["slot_away"]):
            match = re.match(r"Best 3rd \(Groups ([A-L/]+)\)", str(slot))
            if match:
                best_third_slots[slot] = set(match.group(1).split("/"))

        slot_team = {slot: np.full(n_sims, -1) for slot in best_third_slots}
        assignment_cache = {}
        for sim_idx in range(n_sims):
            qualifying_groups = frozenset(groups[best8[k, sim_idx]] for k in range(8))
            assignment = assignment_cache.get(qualifying_groups)
            if assignment is None:
                assignment = self._bipartite(list(qualifying_groups), best_third_slots)
                assignment_cache[qualifying_groups] = assignment
            for slot, group in assignment.items():
                if group is not None:
                    slot_team[slot][sim_idx] = thirds[group][sim_idx]

        ko = {}

        def resolve(slot: str) -> np.ndarray:
            slot = str(slot)
            for pattern, data in [(r"Winner Group ([A-L])", winners), (r"Runner-up Group ([A-L])", runners_up)]:
                match = re.match(pattern, slot)
                if match:
                    return data[match.group(1)]
            if slot in slot_team:
                return slot_team[slot]
            match = re.match(r"Winner Match (\d+)", slot)
            if match:
                return ko[int(match.group(1))]["winner"]
            match = re.match(r"Loser Match (\d+)", slot)
            if match:
                return ko[int(match.group(1))]["loser"]
            raise ValueError(slot)

        for row in self.ks.sort_values("match_id").itertuples(index=False):
            home_idx = resolve(row.slot_home)
            away_idx = resolve(row.slot_away)
            home_goals = rng.poisson(self.LH_KO[home_idx, away_idx])
            away_goals = rng.poisson(self.LA_KO[home_idx, away_idx])
            shootout_home = rng.random(n_sims) < self.PW_KO[home_idx, away_idx]
            winner = np.where(
                home_goals > away_goals,
                home_idx,
                np.where(away_goals > home_goals, away_idx, np.where(shootout_home, home_idx, away_idx)),
            )
            loser = np.where(winner == home_idx, away_idx, home_idx)
            ko[int(row.match_id)] = {
                "home": home_idx,
                "away": away_idx,
                "winner": winner,
                "loser": loser,
                "penalties": home_goals == away_goals,
            }

        n_teams = len(self.keys)

        def pair_mode(data: dict[str, np.ndarray]) -> tuple[int, int]:
            code = data["home"].astype(np.int64) * n_teams + data["away"].astype(np.int64)
            mode = int(np.bincount(code, minlength=n_teams * n_teams).argmax())
            return mode // n_teams, mode % n_teams

        sim = {}
        for match_id, data in ko.items():
            home_idx, away_idx = pair_mode(data)
            sim[match_id] = {
                "home": home_idx,
                "away": away_idx,
                "p_penalties": float(data["penalties"].mean()),
            }
        final_id = int(self.ks[self.ks["round"] == "Final"]["match_id"].iloc[0])
        champion = ko[final_id]["winner"]
        self.champion = sorted(
            ((self.disp[self.keys[idx]], float((champion == idx).mean())) for idx in np.unique(champion)),
            key=lambda item: -item[1],
        )[:8]
        self.sim = sim
        return sim

    def predict_group(self) -> pd.DataFrame:
        rows = []
        for row in self.gf.itertuples(index=False):
            home_key = self.cell_key[row.home_team]
            away_key = self.cell_key[row.away_team]
            home_idx = self.kidx[home_key]
            away_idx = self.kidx[away_key]
            mu_home = self.LH_GROUP[home_idx, away_idx]
            mu_away = self.LA_GROUP[home_idx, away_idx]
            pred_home, pred_away = best_group_score(mu_home, mu_away)
            p_home, p_draw, p_away = poisson_outcome_probs(mu_home, mu_away)
            corners, yellows, reds = self.match_extras(row.match_id, home_key, away_key, ko=False)
            rows.append(
                (
                    pred_home,
                    pred_away,
                    corners,
                    yellows,
                    reds,
                    scoring.outcome_from_probs(p_home, p_draw, p_away),
                )
            )
        out = pd.DataFrame(
            rows,
            columns=[
                "predicted_home_goals",
                "predicted_away_goals",
                "corners",
                "yellow_cards",
                "red_cards",
                "winning_team",
            ],
        )
        return pd.concat([self.gf.reset_index(drop=True), out], axis=1)

    def predict_knockout(self) -> pd.DataFrame:
        if not hasattr(self, "sim"):
            self.simulate()
        predicted_winners = {}
        predicted_losers = {}
        results = {}
        for row in self.ks.sort_values("match_id").itertuples(index=False):
            match_id = int(row.match_id)
            modal = self.sim[match_id]

            def resolve(slot: str, modal_idx: int) -> str:
                winner_match = re.match(r"Winner Match (\d+)", str(slot))
                if winner_match:
                    return predicted_winners[int(winner_match.group(1))]
                loser_match = re.match(r"Loser Match (\d+)", str(slot))
                if loser_match:
                    return predicted_losers[int(loser_match.group(1))]
                return self.keys[modal_idx]

            home_key = resolve(row.slot_home, int(modal["home"]))
            away_key = resolve(row.slot_away, int(modal["away"]))
            home_idx = self.kidx[home_key]
            away_idx = self.kidx[away_key]
            mu_home = self.LH_KO[home_idx, away_idx]
            mu_away = self.LA_KO[home_idx, away_idx]
            pred_home, pred_away = best_knockout_score(mu_home, mu_away)
            p_home, _, p_away = poisson_outcome_probs(mu_home, mu_away)
            penalties = bool(pred_home == pred_away)
            if pred_home > pred_away:
                match_winner = "home"
            elif pred_away > pred_home:
                match_winner = "away"
            else:
                match_winner = self._ko_tie_winner(home_key, away_key, p_home, p_away)
            corners, yellows, reds = self.match_extras(match_id, home_key, away_key, ko=True)
            predicted_winners[match_id] = home_key if match_winner == "home" else away_key
            predicted_losers[match_id] = away_key if match_winner == "home" else home_key
            results[match_id] = (
                self.disp[home_key],
                self.disp[away_key],
                pred_home,
                pred_away,
                corners,
                yellows,
                reds,
                match_winner,
                penalties,
            )

        rows = [results[int(row.match_id)] for row in self.ks.itertuples(index=False)]
        out = pd.DataFrame(
            rows,
            columns=[
                "predicted_home_team",
                "predicted_away_team",
                "predicted_home_goals",
                "predicted_away_goals",
                "corners",
                "yellow_cards",
                "red_cards",
                "match_winner",
                "penalties",
            ],
        )
        return pd.concat([self.ks.reset_index(drop=True), out], axis=1)


def run_submission(
    n_sims: int = 20000,
    verbose: bool = True,
    retrain: bool = True,
) -> tuple[LearnedSubmissionEngine, pd.DataFrame, pd.DataFrame]:
    if retrain:
        train_and_export()
    engine = LearnedSubmissionEngine(n_sims=n_sims, verbose=verbose)
    engine.simulate()
    group_predictions = engine.predict_group()
    knockout_predictions = engine.predict_knockout()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    group_predictions.to_csv(PREDICTIONS / "group_predictions.csv", index=False)
    knockout_predictions.to_csv(PREDICTIONS / "knockout_predictions.csv", index=False)
    group_predictions.to_csv(PREDICTIONS / "group_predictions_learned_weights.csv", index=False)
    knockout_predictions.to_csv(PREDICTIONS / "knockout_predictions_learned_weights.csv", index=False)
    if verbose:
        print("Blended learned-weight title odds:", ", ".join(f"{team} {prob*100:.1f}%" for team, prob in engine.champion))
        print("Wrote predictions/group_predictions.csv and predictions/knockout_predictions.csv")
    return engine, group_predictions, knockout_predictions
