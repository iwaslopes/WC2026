"""Integrated WC2026 engine (v2).

Strength = FIFA ranking points (primary) blended with a results-based Elo. EA FC
ratings are intentionally NOT used. The Poisson goals model is fit on the
engineered match history with sample weights that strongly favour finals
tournaments (World Cups, Euros, Copa, AFCON, ...) over qualifiers and friendlies,
so tournament matchups drive the prediction. Knockout matchups come from a
Monte-Carlo of the whole bracket; a drawn knockout result is sent to penalties.
Runs on numpy + pandas only.
"""
from __future__ import annotations
import math, re, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DATA = ROOT / "data"

# results-based + FIFA features only (no EA FC)
# team-perspective: predict ONE team's goals from its rating gap, its own recent
# form and its opponent's recent form. Neutral tournament -> no home/away term.
FEATURE_COLUMNS = [
    "rating_diff", "rating_sum",                       # FIFA + Elo quality
    "own_recent_gf", "own_recent_ga", "own_recent_win_rate",   # recent form (priority 1)
    "opp_recent_gf", "opp_recent_ga", "opp_recent_win_rate",
    "own_tier_ppg", "opp_tier_ppg",                    # record vs this calibre / FIFA rank (priority 2)
    "own_tour_ppg", "opp_tour_ppg",                    # form in major tournaments (priority 3)
    "importance",
]
FORM_RATES_N = 12        # matches used for each team's recent gf/ga/win-rate

W_FIFA = 0.60          # FIFA-vs-Elo split inside the 'pedigree' component
W_FORM = 0.40          # weight of recent form (the main indicator)
W_TOUR = 0.20          # weight of major-competition performance (below recent form)
TOUR_HL_DAYS = 365 * 4 # gentler half-life: major tournaments are infrequent
TOUR_N = 30            # major-tournament matches scanned
MAIN_COMP_MIN_PHASE = 4.0  # World Cup + continental finals count as 'main'
WC_EXTRA_WEIGHT = 2.0      # World Cup matches weighted higher inside the tournament feature
FORM_N = 20            # recent matches that define current form
FORM_HL_DAYS = 400     # recency half-life within the form window
WC_GOAL_WEIGHT = 0.10  # small attacking nudge from past-World-Cup goals/game
WC_GOALS_SINCE = 2006
# opponent-quality adjustment: how a team does vs opponents of the CURRENT
# opponent's calibre (recency-weighted), shrunk to a low prior when untested.
TIER_ELO_PER_PPG = 90     # elo points per (ppg-vs-tier minus overall ppg)
TIER_BAND = 120           # 'this calibre' = opponents within this elo below (and above) the opp
TIER_HIST_N = 40          # recent matches scanned per team
TIER_PRIOR_PPG = 1.25     # prior ppg vs a strong tier (used when few such matches)
TIER_K = 4.0              # shrinkage strength toward the prior
TIER_ADJ_CLIP = 1.5       # max |ppg gap| applied
RECENCY_HALFLIFE_DAYS = 365 * 1.5   # steep: last ~20 matches dominate; 10y ago ~1%

HOST_COUNTRIES = {"Mexico": "Mexico", "Canada": "Canada", "USA": "United States"}
PLACEHOLDER_TEAMS = {
    "uefa playoff a": "bosnia and herzegovina", "uefa playoff b": "sweden",
    "uefa playoff c": "turkiye", "uefa playoff d": "czechia",
    "fifa playoff 1": "congo dr", "fifa playoff 2": "iraq",
}
TEAM_ALIASES = {
    "united states": "usa", "usa": "usa", "south korea": "korea republic",
    "korea republic": "korea republic", "ivory coast": "cote d'ivoire",
    "cote d'ivoire": "cote d'ivoire", "iran": "ir iran", "ir iran": "ir iran",
    "turkey": "turkiye", "turkiye": "turkiye", "czech republic": "czechia",
    "czechia": "czechia", "dr congo": "congo dr", "congo dr": "congo dr",
    "cape verde": "cabo verde", "cabo verde": "cabo verde", "curacao": "curacao",
    "holland": "netherlands", "netherlands": "netherlands",
}
MAX_GOALS = 8
KO_TIE_THRESHOLD = 0.12   # |P(home win)-P(away win)| below this => coin-flip => penalties
GROUP_TIE_THRESHOLD = 0.16  # group game predicted as a draw when sides are this close
RATING_GAIN = 1.35        # >1 amplifies favourite/underdog gap (more discrepancy). Tunable.
# per-team set-piece / discipline averages (summed for a matchup, then jittered)
CORNER_PER_TEAM_BASE = 4.4   # corners a median team wins per game
CORNER_RATING_COEF = 0.55    # stronger teams win a few more corners
YELLOW_PER_TEAM_BASE = 1.9   # yellows a median team draws per game
YELLOW_RATING_COEF = 0.20    # underdogs foul a little more
# knockout proxy: neutral games between two strong sides ~ tournament KO conditions
KO_PROXY_ELO = 1700
KO_PROXY_BOOST = 3.0
RED_BASE = 0.055         # P(a red card) in a group game (uncommon)
RED_KO_BONUS = 0.04      # knockout games are a bit more heated
N_SIMS = 20000
SEED = 42


def strip_accents(v):
    return "".join(c for c in unicodedata.normalize("NFKD", str(v)) if not unicodedata.combining(c))

def team_key(team):
    k = " ".join(strip_accents(team).strip().lower().replace(".", "").split())
    k = PLACEHOLDER_TEAMS.get(k, k)
    return TEAM_ALIASES.get(k, k)

def venue_country(venue):
    t = str(venue).lower()
    if any(c in t for c in ["azteca", "akron", "bbva", "mexico city", "guadalajara", "monterrey"]):
        return "Mexico"
    if any(c in t for c in ["toronto", "vancouver"]):
        return "Canada"
    return "United States"


def phase_weight(tournament: str) -> float:
    """Big weight for finals tournaments, small for qualifiers / friendlies."""
    t = str(tournament).lower()
    if "world cup" in t and "qual" not in t:
        return 5.0
    if any(k in t for k in ["uefa euro", "copa am", "african cup", "afcon",
                            "afc asian", "asian cup", "gold cup", "confederations"]):
        return 4.0
    if "nations league" in t:
        return 3.0
    if "qual" in t:
        return 3.0          # recent WC qualifiers reflect current form -> weighted
    if "friendly" in t:
        return 1.0
    return 2.0


# ---------- pure-numpy ridge Poisson regression ----------
class RidgePoisson:
    def __init__(self, alpha=0.03):
        self.alpha = alpha
    def fit(self, Xs, y, w, n_iter=100, tol=1e-9):
        n, p = Xs.shape
        X = np.hstack([np.ones((n, 1)), Xs])
        beta = np.zeros(p + 1)
        pen = np.full(p + 1, self.alpha * w.sum() / n); pen[0] = 0.0
        for _ in range(n_iter):
            mu = np.exp(np.clip(X @ beta, -10, 10))
            W = w * mu
            A = (X.T * W) @ X + np.diag(pen)
            g = X.T @ (w * (y - mu)) - pen * beta
            step = np.linalg.solve(A, g)
            beta += step
            if np.max(np.abs(step)) < tol:
                break
        self.beta = beta
        return self
    def predict(self, Xs):
        X = np.hstack([np.ones((len(Xs), 1)), Xs])
        return np.exp(np.clip(X @ self.beta, -10, 10))


class Engine:
    def __init__(self, n_sims=N_SIMS, w_fifa=W_FIFA, gain=RATING_GAIN, verbose=True):
        self.n_sims = n_sims
        self.w_fifa = w_fifa
        self.gain = gain
        self.verbose = verbose
        self._load()
        self._build_ratings()
        self._build_histories()
        self._fit()
        self._precompute_strength_lambdas()

    def _load(self):
        self.strength = pd.read_csv(PROC / "team_strength.csv")
        self.training = pd.read_csv(PROC / "training_matches.csv", parse_dates=["date"])
        self.gf = pd.read_csv(DATA / "group_fixtures.csv")
        self.ks = pd.read_csv(DATA / "knockout_slots.csv")
        self.key_to_team = dict(zip(self.strength.team_key, self.strength.team))
        rr = pd.read_csv(DATA / "external" / "international_results.csv", parse_dates=["date"])
        rr = rr.dropna(subset=["home_score", "away_score"]).sort_values("date")
        rr["hk"] = rr.home_team.map(team_key); rr["ak"] = rr.away_team.map(team_key)
        self.results_raw = rr; self._form_last = rr["date"].max()
        self._fr_cache = {}
        self._hist_cache = {}

    def _build_ratings(self):
        """Effective rating = FIFA points (mapped to Elo scale) blended with Elo."""
        s = self.strength.copy()
        both = s.dropna(subset=["custom_elo", "fifa_points"])
        a, b = np.polyfit(both["fifa_points"], both["custom_elo"], 1)   # fifa -> elo scale
        self.pedigree_by_key = {}
        for r in s.itertuples(index=False):
            elo = r.custom_elo if not pd.isna(r.custom_elo) else np.nan
            fifa_elo = a * r.fifa_points + b if not pd.isna(r.fifa_points) else np.nan
            if not np.isnan(elo) and not np.isnan(fifa_elo):
                rating = self.w_fifa * fifa_elo + (1 - self.w_fifa) * elo
            elif not np.isnan(fifa_elo):
                rating = fifa_elo
            elif not np.isnan(elo):
                rating = elo
            else:
                continue
            self.pedigree_by_key[r.team_key] = float(rating)
        self.rating_by_key = dict(self.pedigree_by_key)   # finalised (with form) in _precompute
        self.attack_mult = {}
        self.rating_median = float(np.median(list(self.pedigree_by_key.values())))
        fp = self.strength.dropna(subset=["fifa_points"])
        self.fifa_by_key = dict(zip(fp.team_key, fp.fifa_points))
        self.fifa_median = float(fp["fifa_points"].median())
        self.fifa_thr = self.fifa_median   # reset to WC-field median in _precompute_strength_lambdas

    def _form_metric(self, key):
        """Recency-weighted (points-per-game + 0.3*goal-diff) over the last
        FORM_N matches. Higher = better current form."""
        r = self.results_raw
        m = r[(r.hk == key) | (r.ak == key)].tail(FORM_N)
        if len(m) < 5:
            return np.nan
        pts, gd, w = [], [], []
        for x in m.itertuples():
            gf, ga = (x.home_score, x.away_score) if x.hk == key else (x.away_score, x.home_score)
            pts.append(3 if gf > ga else 1 if gf == ga else 0); gd.append(gf - ga)
            w.append(0.5 ** ((self._form_last - x.date).days / FORM_HL_DAYS))
        w = np.array(w, float)
        return float(np.average(pts, weights=w) + 0.3 * np.average(gd, weights=w))

    def _tour_metric(self, key):
        """Recency-weighted (ppg + 0.3*goal-diff) in MAIN competitions (World Cup +
        continental finals). Captures big-tournament pedigree, not just recent form."""
        r = self.results_raw
        m = r[(r.hk == key) | (r.ak == key)]
        maj = m[m.tournament.map(phase_weight) >= MAIN_COMP_MIN_PHASE].tail(TOUR_N)
        if len(maj) < 4:
            return np.nan
        pts, gd, w = [], [], []
        for x in maj.itertuples():
            gf, ga = (x.home_score, x.away_score) if x.hk == key else (x.away_score, x.home_score)
            pts.append(3 if gf > ga else 1 if gf == ga else 0); gd.append(gf - ga)
            w.append(0.5 ** ((self._form_last - x.date).days / TOUR_HL_DAYS))
        w = np.array(w, float)
        return float(np.average(pts, weights=w) + 0.3 * np.average(gd, weights=w))

    def rating(self, key):
        return self.rating_by_key.get(key, self.rating_median)

    def _build_histories(self):
        """Per-team chronological history for leak-free as-of features:
        columns = [date_ns, goals_for, goals_against, opponent_rating, is_major]."""
        from collections import defaultdict
        med = self.rating_median; ped = self.pedigree_by_key
        H = defaultdict(list)
        for x in self.results_raw.itertuples(index=False):
            phv = phase_weight(x.tournament)
            H[x.hk].append((x.date.value, x.home_score, x.away_score, ped.get(x.ak, med), phv))
            H[x.ak].append((x.date.value, x.away_score, x.home_score, ped.get(x.hk, med), phv))
        self._hist = {k: np.array(sorted(v), float) for k, v in H.items()}
        self._form_last_ns = self.results_raw["date"].max().value

    def asof_tier(self, key, before, opp_rating):
        """Recency-weighted ppg vs opponents of >= (opp_rating - band), strictly
        before `before` (None = all history), shrunk to a low prior when untested."""
        a = self._hist.get(key)
        if a is None or not len(a):
            return TIER_PRIOR_PPG
        m = a[a[:, 0] < before] if before is not None else a
        if not len(m):
            return TIER_PRIOR_PPG
        ref = before if before is not None else self._form_last_ns
        sel = m[m[:, 3] >= opp_rating - TIER_BAND]
        if not len(sel):
            return TIER_PRIOR_PPG
        w = 0.5 ** ((ref - sel[:, 0]) / (FORM_HL_DAYS * 86400e9))
        pts = np.where(sel[:, 1] > sel[:, 2], 3.0, np.where(sel[:, 1] == sel[:, 2], 1.0, 0.0))
        n = w.sum() / max(w.mean(), 1e-9)
        return float((n * np.average(pts, weights=w) + TIER_K * TIER_PRIOR_PPG) / (n + TIER_K))

    def asof_tour(self, key, before, own_rating):
        """Major-tournament performance vs Elo expectation ("tournament surprise"):
        recency-weighted mean of (actual result - Elo-expected result) over World
        Cup / continental-final matches, World Cup games weighted higher. Positive
        => the team over-performs its rating when it matters; ~0 => as expected."""
        a = self._hist.get(key)
        if a is None or not len(a):
            return 0.0
        m = a[a[:, 0] < before] if before is not None else a
        maj = m[m[:, 4] >= MAIN_COMP_MIN_PHASE]
        if len(maj) < 4:
            return 0.0
        ref = before if before is not None else self._form_last_ns
        e = 1.0 / (1.0 + 10 ** (-(own_rating - maj[:, 3]) / 400.0))      # Elo-expected result
        res = np.where(maj[:, 1] > maj[:, 2], 1.0, np.where(maj[:, 1] == maj[:, 2], 0.5, 0.0))
        wc = np.where(maj[:, 4] >= 5.0, WC_EXTRA_WEIGHT, 1.0)
        w = 0.5 ** ((ref - maj[:, 0]) / (TOUR_HL_DAYS * 86400e9)) * wc
        return float(np.average(res - e, weights=w))

    @staticmethod
    def _team_rows(tr):
        """Two team-perspective feature blocks (home-team-scoring, away-team-scoring)."""
        def block(own, opp):
            return np.column_stack([
                tr[f"{own}_elo"] - tr[f"{opp}_elo"],          # rating_diff
                tr[f"{own}_elo"] + tr[f"{opp}_elo"],          # rating_sum
                tr[f"{own}_recent_gf"], tr[f"{own}_recent_ga"], tr[f"{own}_recent_win_rate"],
                tr[f"{opp}_recent_gf"], tr[f"{opp}_recent_ga"], tr[f"{opp}_recent_win_rate"],
                tr[f"{own}_tier_ppg"], tr[f"{opp}_tier_ppg"],
                tr[f"{own}_tour_ppg"], tr[f"{opp}_tour_ppg"],
                tr["importance"],
            ]).astype(float)
        Xh = block("home", "away"); Xa = block("away", "home")
        return Xh, Xa

    def _fit(self):
        tr = self.training.sort_values("date").reset_index(drop=True)
        # leak-free as-of tier (vs opponent calibre) and tournament-form features
        med = self.rating_median; ped = self.pedigree_by_key
        dvals = tr["date"].astype("int64").values
        hk = tr["home_key"].values; ak = tr["away_key"].values
        n = len(tr)
        htier = np.empty(n); atier = np.empty(n); htour = np.empty(n); atour = np.empty(n)
        for i in range(n):
            d = dvals[i]
            htier[i] = self.asof_tier(hk[i], d, ped.get(ak[i], med))
            atier[i] = self.asof_tier(ak[i], d, ped.get(hk[i], med))
            htour[i] = self.asof_tour(hk[i], d, ped.get(hk[i], med))
            atour[i] = self.asof_tour(ak[i], d, ped.get(ak[i], med))
        tr["home_tier_ppg"] = htier; tr["away_tier_ppg"] = atier
        tr["home_tour_ppg"] = htour; tr["away_tour_ppg"] = atour
        last = tr["date"].max()
        recency = 0.5 ** ((last - tr["date"]).dt.days.clip(lower=0) / RECENCY_HALFLIFE_DAYS)
        phase = tr["tournament"].map(phase_weight)
        ko_proxy = np.where((tr["neutral"] == 1) & (tr["home_elo"] > KO_PROXY_ELO)
                            & (tr["away_elo"] > KO_PROXY_ELO), KO_PROXY_BOOST, 1.0)
        w = (recency * phase * ko_proxy).values
        Xh, Xa = self._team_rows(tr)
        yh, ya = tr["home_goals"].values.astype(float), tr["away_goals"].values.astype(float)
        Xall = np.vstack([Xh, Xa])
        self.mean_, self.std_ = Xall.mean(0), Xall.std(0); self.std_[self.std_ == 0] = 1.0
        std = lambda M: (M - self.mean_) / self.std_
        cut = tr["date"].quantile(0.85); m = (tr["date"] < cut).values
        # validation: train on early matches, score the recent hold-out (one MAE)
        Xtr = np.vstack([Xh[m], Xa[m]]); ytr = np.concatenate([yh[m], ya[m]]); wtr = np.concatenate([w[m], w[m]])
        vm = RidgePoisson().fit(std(Xtr), ytr, wtr)
        lamH = np.clip(vm.predict(std(Xh[~m])), .05, 6); lamA = np.clip(vm.predict(std(Xa[~m])), .05, 6)
        mae = float(np.mean(np.abs(np.concatenate([lamH - yh[~m], lamA - ya[~m]]))))
        exact = float(np.mean([self.best_score(h, a)[:2] == (int(a1), int(a2))
                               for h, a, a1, a2 in zip(lamH, lamA, yh[~m], ya[~m])]))
        self.metrics = dict(validation_rows=int((~m).sum()), validation_start=str(pd.Timestamp(cut).date()),
                            goal_mae=mae, exact_score_rate_mode=exact)
        # final model on all team-perspective rows
        self.goal_model = RidgePoisson().fit(std(Xall), np.concatenate([yh, ya]), np.concatenate([w, w]))
        # learned weights (standardised coefficients = comparable importance)
        self.learned_weights = dict(zip(FEATURE_COLUMNS, self.goal_model.beta[1:]))
        if self.verbose:
            print("Symmetric goals model — validation:",
                  {k: round(v, 3) if isinstance(v, float) else v for k, v in self.metrics.items()})
            grp = {
                "recent form": sum(abs(self.learned_weights[c]) for c in
                    ["own_recent_gf","own_recent_ga","own_recent_win_rate","opp_recent_gf","opp_recent_ga","opp_recent_win_rate"]),
                "h2h vs rank (tier)": sum(abs(self.learned_weights[c]) for c in ["own_tier_ppg","opp_tier_ppg"]),
                "tournament form": sum(abs(self.learned_weights[c]) for c in ["own_tour_ppg","opp_tour_ppg"]),
                "FIFA/Elo rating": abs(self.learned_weights["rating_diff"]),
            }
            tot = sum(grp.values()) or 1.0
            print("Learned variable importance:", {k: f"{v/tot*100:.0f}%" for k, v in sorted(grp.items(), key=lambda x:-x[1])})

    # ---- feature construction ----
    def _team_hist(self, key):
        """Recent matches for a team: (points, opponent_rating, recency_weight)."""
        if key in self._hist_cache:
            return self._hist_cache[key]
        r = self.results_raw
        m = r[(r.hk == key) | (r.ak == key)].tail(TIER_HIST_N)
        rows = []
        for x in m.itertuples():
            gf, ga = (x.home_score, x.away_score) if x.hk == key else (x.away_score, x.home_score)
            opp = x.ak if x.hk == key else x.hk
            pts = 3 if gf > ga else 1 if gf == ga else 0
            opp_r = self.pedigree_by_key.get(opp, self.rating_median)
            w = 0.5 ** ((self._form_last - x.date).days / FORM_HL_DAYS)
            rows.append((pts, opp_r, w))
        a = np.array(rows, float) if rows else np.zeros((0, 3))
        self._hist_cache[key] = a
        return a

    def _ppg_overall(self, key):
        a = self._team_hist(key)
        return float(np.average(a[:, 0], weights=a[:, 2])) if len(a) else 1.4

    def _ppg_vs_tier(self, key, opp_rating):
        """Recency-weighted ppg vs opponents of >= (opp_rating - band), shrunk to a
        low prior so untested teams (e.g. strong form only vs weak sides) are not
        credited against top opposition."""
        a = self._team_hist(key)
        if not len(a):
            return TIER_PRIOR_PPG
        sel = a[a[:, 1] >= opp_rating - TIER_BAND]
        n = sel[:, 2].sum() / max(a[:, 2].mean(), 1e-9) if len(sel) else 0.0
        obs = float(np.average(sel[:, 0], weights=sel[:, 2])) if len(sel) else TIER_PRIOR_PPG
        return (n * obs + TIER_K * TIER_PRIOR_PPG) / (n + TIER_K)

    def _form_rates(self, key):
        """Venue-agnostic recent form: mean goals for/against and win rate over a
        team's last FORM_RATES_N matches."""
        if key in self._fr_cache:
            return self._fr_cache[key]
        r = self.results_raw
        m = r[(r.hk == key) | (r.ak == key)].tail(FORM_RATES_N)
        if len(m) < 4:
            v = (1.25, 1.25, 1 / 3)
        else:
            gf = np.where(m.hk.values == key, m.home_score.values, m.away_score.values)
            ga = np.where(m.hk.values == key, m.away_score.values, m.home_score.values)
            wr = np.mean(gf > ga)
            v = (float(gf.mean()), float(ga.mean()), float(wr))
        self._fr_cache[key] = v
        return v

    def _features(self, own_key, opp_key):
        ro, rp = self.rating(own_key), self.rating(opp_key)   # FIFA + Elo pedigree
        ogf, oga, owr = self._form_rates(own_key)
        pgf, pga, pwr = self._form_rates(opp_key)
        return {
            "rating_diff": ro - rp, "rating_sum": ro + rp,
            "own_recent_gf": ogf, "own_recent_ga": oga, "own_recent_win_rate": owr,
            "opp_recent_gf": pgf, "opp_recent_ga": pga, "opp_recent_win_rate": pwr,
            "own_tier_ppg": self.asof_tier(own_key, None, rp),   # record vs opp's calibre
            "opp_tier_ppg": self.asof_tier(opp_key, None, ro),
            "own_tour_ppg": self.asof_tour(own_key, None, ro),   # tournament form vs Elo expectation
            "opp_tour_ppg": self.asof_tour(opp_key, None, rp),
            "importance": 5.0,
        }

    def lambdas(self, home_key, away_key, home_adv=0):
        """Neutral-venue expected goals: one symmetric goals model applied from
        each team's perspective. home_adv is ignored (World Cup is neutral)."""
        def pred(own, opp):
            x = np.array([[self._features(own, opp)[c] for c in FEATURE_COLUMNS]], float)
            return self.goal_model.predict((x - self.mean_) / self.std_)[0]
        lam_h = pred(home_key, away_key) * self.attack_mult.get(home_key, 1.0)
        lam_a = pred(away_key, home_key) * self.attack_mult.get(away_key, 1.0)
        return float(np.clip(lam_h, .05, 6)), float(np.clip(lam_a, .05, 6))

    def _precompute_strength_lambdas(self):
        cells = sorted(set(self.gf.home_team) | set(self.gf.away_team))
        self.cell_key = {c: team_key(c) for c in cells}
        keys = sorted(set(self.cell_key.values()))
        self.keys = keys
        self.kidx = {k: i for i, k in enumerate(keys)}
        self.disp = {k: self.key_to_team.get(k, k.title()) for k in keys}
        n = len(keys)
        self.fifa_thr = float(np.median([self._fifa(k) for k in keys]))
        # already-played results (overwrite predictions with the real score)
        self.played = {}
        pf = DATA / "wc2026_results_so_far.csv"
        if pf.exists():
            rdf = pd.read_csv(pf)
            rmap = {}
            for x in rdf.itertuples(index=False):
                rmap[frozenset({team_key(x.team1), team_key(x.team2)})] = (team_key(x.team1), int(x.team1_score), int(x.team2_score))
            for r in self.gf.itertuples(index=False):
                hk = self.cell_key[r.home_team]; ak = self.cell_key[r.away_team]
                rec = rmap.get(frozenset({hk, ak}))
                if rec:
                    t1k, s1, s2 = rec
                    self.played[int(r.match_id)] = (s1, s2) if t1k == hk else (s2, s1)
        # strength rating = FIFA + Elo pedigree only; form / tier / tournament now
        # enter the goals model as separate LEARNED features (see _fit / _features).
        self.rating_by_key = dict(self.pedigree_by_key)
        # small World-Cup scoring-pedigree nudge to attack (low weight)
        wcm = self.results_raw[
            self.results_raw.tournament.str.contains("FIFA World Cup", case=False)
            & ~self.results_raw.tournament.str.contains("qual", case=False)
            & (self.results_raw.date.dt.year >= WC_GOALS_SINCE)]
        gpg = {}
        for k in keys:
            m = wcm[(wcm.hk == k) | (wcm.ak == k)]
            if len(m) >= 3:
                gs = np.where(m.hk.values == k, m.home_score.values, m.away_score.values).sum()
                gpg[k] = gs / len(m)
        if gpg:
            gmean = float(np.mean(list(gpg.values())))
            for k in gpg:
                self.attack_mult[k] = float(np.clip(
                    1 + WC_GOAL_WEIGHT * (gpg[k] - gmean) / gmean, 0.92, 1.10))
        # per-team set-piece / discipline medians (each team its own value)
        rat = np.array([self.rating(k) for k in keys]); mu = rat.mean(); sd = rat.std() or 1.0
        self.corner_avg, self.yellow_avg = {}, {}
        for k in keys:
            z = (self.rating(k) - mu) / sd
            self.corner_avg[k] = CORNER_PER_TEAM_BASE + CORNER_RATING_COEF * z
            self.yellow_avg[k] = YELLOW_PER_TEAM_BASE - YELLOW_RATING_COEF * z
        try:
            pd.DataFrame({"team": [self.disp[k] for k in keys],
                          "corner_avg_per_game": [round(self.corner_avg[k], 2) for k in keys],
                          "yellow_avg_per_game": [round(self.yellow_avg[k], 2) for k in keys]}
                         ).to_csv(ROOT / "config" / "team_set_pieces.csv", index=False)
        except Exception:
            pass
        self.LH = np.zeros((n, n)); self.LA = np.zeros((n, n))
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i == j: continue
                self.LH[i, j], self.LA[i, j] = self.lambdas(ki, kj, 0)

    # ---- scoring ----
    @staticmethod
    def _pmf(rate, kmax=MAX_GOALS):
        k = np.arange(kmax + 1)
        p = np.exp(-rate) * rate ** k / np.array([math.factorial(int(i)) for i in k])
        return p / p.sum()
    def score_mat(self, lh, la):
        return np.outer(self._pmf(lh), self._pmf(la))
    def best_score(self, lh, la, include_winner=True):
        P = self.score_mat(lh, la); n = P.shape[0]
        ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        best, bev = (0, 0), -1.0
        for pi in range(n):
            for pj in range(n):
                exact = P[pi, pj] * 25
                gd = ((pi - pj) == (ai - aj)) & ~((ai == pi) & (aj == pj))
                tot = ((pi + pj) == (ai + aj)) & ~((ai == pi) & (aj == pj))
                ev = exact + 10 * (P[gd].sum() + P[tot].sum())
                if include_winner:
                    po = 0 if pi > pj else (1 if pi < pj else 2)
                    out = np.where(ai > aj, 0, np.where(ai < aj, 1, 2))
                    ev += 40 * P[out == po].sum()
                if ev > bev:
                    bev, best = ev, (pi, pj)
        return best[0], best[1], bev
    @staticmethod
    def outcome_probs(P):
        n = P.shape[0]
        return (P[np.tril_indices(n, -1)].sum(), np.trace(P), P[np.triu_indices(n, 1)].sum())

    def best_score_ko(self, lh, la):
        """KO scoreline by JOINT expected value of the score field AND the
        penalties field (penalties=True iff the predicted result is a draw).
        A decisive score is only beaten by a draw when a draw is genuinely
        likely, so shootouts are predicted only when probable."""
        P = self.score_mat(lh, la); n = P.shape[0]
        p_home, p_draw, p_away = self.outcome_probs(P)
        draw_modal = abs(p_home - p_away) < KO_TIE_THRESHOLD   # coin-flip -> goes to penalties
        fav_home = p_home >= p_away
        ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        best, bev = (1, 1) if draw_modal else (1, 0), -1.0
        for pi in range(n):
            for pj in range(n):
                if draw_modal and pi != pj:
                    continue
                if not draw_modal:                       # restrict to the favoured side winning
                    if fav_home and not pi > pj: continue
                    if not fav_home and not pj > pi: continue
                exact = P[pi, pj] * 25
                gd = ((pi - pj) == (ai - aj)) & ~((ai == pi) & (aj == pj))
                tot = ((pi + pj) == (ai + aj)) & ~((ai == pi) & (aj == pj))
                ev = exact + 10 * (P[gd].sum() + P[tot].sum())
                if ev > bev:
                    bev, best = ev, (pi, pj)
        return best[0], best[1]

    def _fifa(self, key):
        return self.fifa_by_key.get(key, self.fifa_median)

    def match_extras(self, match_id, home_key, away_key, ko):
        """Per-team medians summed for the matchup, plus match-specific jitter.
        Corners = corner_avg[home] + corner_avg[away]; yellows likewise (+ a small
        knockout bump). Jitter is seeded by match_id, so values vary every match
        yet the submission is reproducible. No card/corner data exists in the
        source, so the per-team medians are derived from team strength."""
        rng = np.random.default_rng([SEED, int(match_id)])   # SeedSequence mixes well
        corners = self.corner_avg.get(home_key, CORNER_PER_TEAM_BASE) + \
                  self.corner_avg.get(away_key, CORNER_PER_TEAM_BASE) + rng.normal(0, 1.1)
        yellows = self.yellow_avg.get(home_key, YELLOW_PER_TEAM_BASE) + \
                  self.yellow_avg.get(away_key, YELLOW_PER_TEAM_BASE) + \
                  (0.7 if ko else 0.0) + rng.normal(0, 0.9)
        red_p = RED_BASE + (RED_KO_BONUS if ko else 0.0)   # uncommon, but happens
        rr = rng.random()
        reds = (1 if rr < red_p else 0) + (1 if rr < red_p * 0.12 else 0)
        return int(round(np.clip(corners, 3, 17))), int(round(np.clip(yellows, 1, 9))), reds

    # ---------------- Monte-Carlo bracket ----------------
    def _bipartite(self, groups, slots_allowed):
        s2g = {s: None for s in slots_allowed}
        def assign(g, seen):
            for s, al in slots_allowed.items():
                if g in al and s not in seen:
                    seen.add(s)
                    if s2g[s] is None or assign(s2g[s], seen):
                        s2g[s] = g; return True
            return False
        for g in groups: assign(g, set())
        return s2g

    def simulate(self):
        rng = np.random.default_rng(SEED); n = self.n_sims
        gf, ks = self.gf, self.ks
        groups = sorted(gf.group.unique())
        rating = np.array([self.rating(k) for k in self.keys])
        gteams = {g: sorted({self.kidx[self.cell_key[c]]
                             for c in set(gf[gf.group == g].home_team) | set(gf[gf.group == g].away_team)})
                  for g in groups}
        stats = {g: np.zeros((len(gteams[g]), 3, n)) for g in groups}
        for g in groups:
            loc = {t: i for i, t in enumerate(gteams[g])}
            for r in gf[gf.group == g].itertuples(index=False):
                hi = self.kidx[self.cell_key[r.home_team]]; ai = self.kidx[self.cell_key[r.away_team]]
                lh, la = self.lambdas(self.keys[hi], self.keys[ai], 0)   # neutral venue
                hg = rng.poisson(lh, n); ag = rng.poisson(la, n)
                hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0)); ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
                lh_, la_ = loc[hi], loc[ai]
                stats[g][lh_, 0] += hp; stats[g][la_, 0] += ap
                stats[g][lh_, 1] += hg - ag; stats[g][la_, 1] += ag - hg
                stats[g][lh_, 2] += hg; stats[g][la_, 2] += ag
        win, run, third, third_st = {}, {}, {}, {}
        for g in groups:
            st = stats[g]; arr = np.array(gteams[g])
            sc = st[:, 0]*1e6 + st[:, 1]*1e3 + st[:, 2] + rng.random((len(arr), n))*1e-3
            order = np.argsort(-sc, axis=0)
            win[g], run[g], third[g] = arr[order[0]], arr[order[1]], arr[order[2]]
            rows = order[2]
            third_st[g] = np.stack([st[rows, 0, np.arange(n)], st[rows, 1, np.arange(n)], st[rows, 2, np.arange(n)]])
        tsc = np.stack([third_st[g][0]*1e6 + third_st[g][1]*1e3 + third_st[g][2] + rng.random(n)*1e-3 for g in groups])
        best8 = np.argsort(-tsc, axis=0)[:8]
        b3 = {}
        for s in set(ks.slot_home) | set(ks.slot_away):
            m = re.match(r"Best 3rd \(Groups ([A-L/]+)\)", str(s))
            if m: b3[s] = set(m.group(1).split("/"))
        slot_team = {s: np.full(n, -1) for s in b3}; cache = {}
        for sim in range(n):
            qual = frozenset(groups[best8[k, sim]] for k in range(8))
            a = cache.get(qual)
            if a is None: a = self._bipartite(list(qual), b3); cache[qual] = a
            for s, g in a.items():
                if g is not None: slot_team[s][sim] = third[g][sim]
        ko = {}
        def resolve(slot):
            slot = str(slot)
            for pat, d in [(r"Winner Group ([A-L])", win), (r"Runner-up Group ([A-L])", run)]:
                m = re.match(pat, slot)
                if m: return d[m.group(1)]
            if slot in slot_team: return slot_team[slot]
            m = re.match(r"Winner Match (\d+)", slot)
            if m: return ko[int(m.group(1))]["w"]
            m = re.match(r"Loser Match (\d+)", slot)
            if m: return ko[int(m.group(1))]["l"]
            raise ValueError(slot)
        for r in ks.sort_values("match_id").itertuples(index=False):
            hi = resolve(r.slot_home); ai = resolve(r.slot_away)
            lh = self.LH[hi, ai]; la = self.LA[hi, ai]
            hg = rng.poisson(lh); ag = rng.poisson(la)
            ph = 1.0 / (1.0 + 10 ** ((rating[ai] - rating[hi]) / 400.0))
            coin = rng.random(n) < ph
            w = np.where(hg > ag, hi, np.where(ag > hg, ai, np.where(coin, hi, ai)))
            l = np.where(w == hi, ai, hi)
            pens = (hg == ag)                       # drawn after ET -> shootout
            ko[int(r.match_id)] = dict(h=hi, a=ai, w=w, l=l, pens=pens)
        nteam = len(self.keys)
        def pair_mode(d):
            code = d["h"].astype(np.int64) * nteam + d["a"].astype(np.int64)
            c = int(np.bincount(code, minlength=nteam * nteam).argmax())
            return c // nteam, c % nteam
        out = {}
        for mid, d in ko.items():
            h, a = pair_mode(d)
            out[mid] = dict(home=h, away=a, p_pens=float(d["pens"].mean()))
        fmid = int(ks[ks["round"] == "Final"]["match_id"].iloc[0])
        champ = ko[fmid]["w"]
        cp = sorted(((self.disp[self.keys[t]], float((champ == t).mean())) for t in np.unique(champ)),
                    key=lambda x: -x[1])[:8]
        self.sim = out; self.champion = cp
        return out

    # ---------------- prediction tables ----------------
    def predict_group(self):
        rows = []
        for r in self.gf.itertuples(index=False):
            hk = self.cell_key[r.home_team]; ak = self.cell_key[r.away_team]
            corners, yellows, reds = self.match_extras(r.match_id, hk, ak, ko=False)
            mid = int(r.match_id)
            if mid in self.played:                       # already played -> actual score
                pi, pj = self.played[mid]
                wt = "home" if pi > pj else ("away" if pi < pj else "draw")
            else:
                lh, la = self.lambdas(hk, ak, 0)         # neutral venue
                P = self.score_mat(lh, la); ph, pd_, pa = self.outcome_probs(P)
                if abs(ph - pa) < GROUP_TIE_THRESHOLD:   # close -> realistic draw
                    d = int(np.argmax(np.diag(P))); pi = pj = d; wt = "draw"
                else:
                    pi, pj, _ = self.best_score(lh, la, include_winner=True)
                    wt = "home" if pi > pj else ("away" if pi < pj else "draw")
            rows.append((pi, pj, corners, yellows, reds, wt))
        out = pd.DataFrame(rows, columns=["predicted_home_goals", "predicted_away_goals",
                                          "corners", "yellow_cards", "red_cards", "winning_team"])
        gf2 = self.gf.reset_index(drop=True).copy()
        gf2["home_team"] = gf2["home_team"].map(lambda c: self.disp[self.cell_key[c]])
        gf2["away_team"] = gf2["away_team"].map(lambda c: self.disp[self.cell_key[c]])
        return pd.concat([gf2, out], axis=1)

    def predict_knockout(self):
        """Build ONE consistent bracket: R32 matchups come from the simulation's
        most-likely occupants; from the round of 16 on, each slot is filled by the
        predicted winner / loser of its feeder match. This guarantees the
        third-place game holds the two semifinal LOSERS, not a finalist."""
        if not hasattr(self, "sim"): self.simulate()
        pw, pl, results = {}, {}, {}
        for r in self.ks.sort_values("match_id").itertuples(index=False):
            mid = int(r.match_id); d = self.sim[mid]
            def resolve(slot, modal_idx):
                m = re.match(r"Winner Match (\d+)", str(slot))
                if m: return pw[int(m.group(1))]
                m = re.match(r"Loser Match (\d+)", str(slot))
                if m: return pl[int(m.group(1))]
                return self.keys[modal_idx]            # group-fed slot -> simulation modal
            hk = resolve(r.slot_home, d["home"]); ak = resolve(r.slot_away, d["away"])
            hi, ai = self.kidx[hk], self.kidx[ak]
            lh, la = self.LH[hi, ai], self.LA[hi, ai]
            pi, pj = self.best_score_ko(lh, la)
            ph, pd_, pa = self.outcome_probs(self.score_mat(lh, la))
            corners, yellows, reds = self.match_extras(mid, hk, ak, ko=True)
            penalties = bool(pi == pj)
            mw = ("home" if pi > pj else "away" if pj > pi else ("home" if ph >= pa else "away"))
            pw[mid] = hk if mw == "home" else ak
            pl[mid] = ak if mw == "home" else hk
            results[mid] = (self.disp[hk], self.disp[ak], pi, pj, corners, yellows, reds, mw, penalties)
        rows = [results[int(r.match_id)] for r in self.ks.itertuples(index=False)]
        out = pd.DataFrame(rows, columns=["predicted_home_team", "predicted_away_team",
                                          "predicted_home_goals", "predicted_away_goals",
                                          "corners", "yellow_cards", "red_cards",
                                          "match_winner", "penalties"])
        return pd.concat([self.ks.reset_index(drop=True), out], axis=1)


def run_all(n_sims=N_SIMS, verbose=True):
    eng = Engine(n_sims=n_sims, verbose=verbose)
    eng.simulate()
    g = eng.predict_group(); k = eng.predict_knockout()
    if verbose:
        print("Title odds:", ", ".join(f"{t} {p*100:.1f}%" for t, p in eng.champion))
    return eng, g, k
