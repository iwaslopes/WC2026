"""The competition's scoring rubric, plus the routine that turns a score
probability matrix into the scoreline that maximizes EXPECTED points (not the
single most likely score). This is what makes the predictions rubric-optimal:
the partial-credit rules (correct goal-difference OR correct total = 10) pull
the pick toward common low scores.
"""
import numpy as np


def score_points(pi, pj, ai, aj):
    """Points for the SCORE category only (max 25)."""
    if pi == ai and pj == aj:
        return 25
    if (pi - pj) == (ai - aj):      # correct goal difference, wrong score
        return 10
    if (pi + pj) == (ai + aj):      # correct total goals, wrong score
        return 10
    return 0


def optimal_scoreline(M, max_pick=6):
    """Return (i, j) maximizing expected SCORE points under M."""
    n = M.shape[0]
    ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best, best_ev = (1, 0), -1.0
    for pi in range(max_pick + 1):
        for pj in range(max_pick + 1):
            exact = M[pi, pj] * 25
            gd = ((pi - pj) == (ai - aj)) & ~((ai == pi) & (aj == pj))
            tot = ((pi + pj) == (ai + aj)) & ~((ai == pi) & (aj == pj))
            ev = exact + 10 * (M[gd].sum() + M[tot].sum())
            if ev > best_ev:
                best_ev, best = ev, (pi, pj)
    return best, best_ev


def outcome_from_probs(p_home, p_draw, p_away):
    """Group-stage winning_team: 'home' / 'away' / 'draw' (40 pts)."""
    return max((p_home, "home"), (p_draw, "draw"), (p_away, "away"))[1]


def ko_winner(p_home, p_away):
    """Knockout match_winner: 'home' / 'away' (draws resolved in ET/pens)."""
    return "home" if p_home >= p_away else "away"
