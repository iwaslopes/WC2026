"""Load results.csv and build the per-team *effective rating* that the goals
model consumes. The effective rating is a blend of three strength signals:

    Elo  (learned from full match history, primary)
    FIFA points  (current ranking, secondary)
    FM26 strength (Football Manager 26 squad strength, optional)

FIFA points and FM26 are mapped onto the Elo scale by a linear fit across teams,
then blended with weights config.W_FIFA / config.W_FM. Those weights are the
"per-variable" weights and can be calibrated (see run.calibrate).
"""
import numpy as np
import pandas as pd
from . import config, teams


def load_results() -> pd.DataFrame:
    df = pd.read_csv(config.RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def _linear_map(x, y):
    """Least-squares y ~ m*x + c; returns predictor f(x)."""
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


def build_effective_ratings(latest_elo, feat_df, wc_team_keys,
                            w_fifa=None, w_fm=None):
    """wc_team_keys: dict {results_name -> normalized_feature_key} for WC teams.
    Returns {results_name -> effective_rating(float)}.
    """
    w_fifa = config.W_FIFA if w_fifa is None else w_fifa
    w_fm = config.W_FM if w_fm is None else w_fm

    names = list(wc_team_keys)
    elo = np.array([latest_elo.get(n, config.ELO_START) for n in names])
    fifa = np.array([feat_df["fifa_points"].get(wc_team_keys[n], np.nan) for n in names], float)
    fm = np.array([feat_df["fm26_strength"].get(wc_team_keys[n], np.nan) for n in names], float)

    # map FIFA / FM onto Elo scale using teams that have both
    fifa_elo = np.full_like(elo, np.nan)
    if np.isfinite(fifa).sum() >= 3:
        mask = np.isfinite(fifa)
        f = _linear_map(fifa[mask], elo[mask])
        fifa_elo[mask] = f(fifa[mask])
    fm_elo = np.full_like(elo, np.nan)
    if np.isfinite(fm).sum() >= 3:
        mask = np.isfinite(fm)
        f = _linear_map(fm[mask], elo[mask])
        fm_elo[mask] = f(fm[mask])

    out = {}
    for i, n in enumerate(names):
        parts, wts = [elo[i]], [max(1.0 - w_fifa - w_fm, 0.0)]
        if np.isfinite(fifa_elo[i]) and w_fifa > 0:
            parts.append(fifa_elo[i]); wts.append(w_fifa)
        if np.isfinite(fm_elo[i]) and w_fm > 0:
            parts.append(fm_elo[i]); wts.append(w_fm)
        wts = np.array(wts) / sum(wts)
        out[n] = float(np.dot(wts, parts))
    return out
