"""Bivariate-Poisson (Dixon-Coles) goals model, implemented in pure numpy.

  log(mu_team) = b0 + b_rating * (rating_gap/100) + b_home * is_home

`rating_gap` is the *effective rating* difference (Elo blended with FIFA / FM26 -
see features.py). The fitted coefficients b_rating and b_home are the learned
weights mapping strength and home advantage to goals. A single Dixon-Coles `rho`
corrects the dependence in low-scoring lines (0-0, 1-0, 0-1, 1-1).
"""
import math
import numpy as np
from . import config

_LGAMMA = np.array([math.lgamma(k + 1) for k in range(config.MAX_GOALS + 2)])


def _pois_pmf(mu, kmax):
    k = np.arange(kmax + 1)
    return np.exp(k * np.log(max(mu, 1e-9)) - mu - _LGAMMA[:kmax + 1])


class PoissonGLM:
    """Weighted Poisson regression via IRLS (no scipy needed)."""

    def __init__(self):
        self.beta = None

    def fit(self, X, y, w, n_iter=50, tol=1e-8):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.asarray(w, float)
        beta = np.zeros(X.shape[1])
        for _ in range(n_iter):
            eta = np.clip(X @ beta, -10, 10)
            mu = np.exp(eta)
            Wd = w * mu
            z = eta + (y - mu) / np.maximum(mu, 1e-9)
            XtW = X.T * Wd
            A = XtW @ X + 1e-6 * np.eye(X.shape[1])   # tiny ridge for stability
            b = XtW @ z
            new = np.linalg.solve(A, b)
            if np.max(np.abs(new - beta)) < tol:
                beta = new
                break
            beta = new
        self.beta = beta
        return self

    def mu(self, rating_gap, is_home):
        x = np.array([1.0, rating_gap / 100.0, float(is_home)])
        return float(np.exp(np.clip(x @ self.beta, -10, 10)))


def build_training(train_df):
    """Two rows per match (one per team perspective)."""
    X, y, w = [], [], []
    last = train_df["date"].max()
    hl = config.RECENCY_HALFLIFE_DAYS
    for r in train_df.itertuples(index=False):
        age = (last - r.date).days
        wt = 0.5 ** (age / hl)
        gap = r.elo_home_pre - r.elo_away_pre
        home_flag = 0.0 if r.neutral else 1.0
        # home team scores home_score
        X.append([1.0, gap / 100.0, home_flag]); y.append(r.home_score); w.append(wt)
        # away team scores away_score
        X.append([1.0, -gap / 100.0, 0.0]);      y.append(r.away_score); w.append(wt)
    return np.array(X), np.array(y), np.array(w)


def _dc_tau(i, j, mu_h, mu_a, rho):
    if i == 0 and j == 0:
        return 1.0 - mu_h * mu_a * rho
    if i == 0 and j == 1:
        return 1.0 + mu_h * rho
    if i == 1 and j == 0:
        return 1.0 + mu_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(mu_h, mu_a, rho=0.0, kmax=None):
    kmax = kmax or config.MAX_GOALS
    ph = _pois_pmf(mu_h, kmax)
    pa = _pois_pmf(mu_a, kmax)
    M = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= _dc_tau(i, j, mu_h, mu_a, rho)
    M = np.clip(M, 1e-12, None)
    return M / M.sum()


def fit_rho(train_df, model, grid=None):
    """1-D MLE for the Dixon-Coles dependence parameter on low scores."""
    grid = grid if grid is not None else np.linspace(-0.2, 0.2, 41)
    rows = []
    for r in train_df.itertuples(index=False):
        gap = r.elo_home_pre - r.elo_away_pre
        hf = 0.0 if r.neutral else 1.0
        mh = model.mu(gap, hf)
        ma = model.mu(-gap, 0.0)
        rows.append((min(r.home_score, 1), min(r.away_score, 1), mh, ma,
                     int(r.home_score == 0 or r.home_score == 1),
                     int(r.away_score == 0 or r.away_score == 1)))
    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        ll = 0.0
        for hi, aj, mh, ma, lh, la in rows:
            if lh and la:                      # only low-score cells carry the correction
                ll += math.log(max(_dc_tau(hi, aj, mh, ma, rho), 1e-9))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)


def outcome_probs(M):
    n = M.shape[0]
    idx = np.arange(n)
    home = M[np.tril_indices(n, -1)].sum()      # i>j
    away = M[np.triu_indices(n, 1)].sum()        # j>i
    draw = np.trace(M)
    eg_h = (M.sum(axis=1) * idx).sum()
    eg_a = (M.sum(axis=0) * idx).sum()
    return home, draw, away, eg_h, eg_a
