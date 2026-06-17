"""Monte-Carlo tournament simulator.

Group stage -> standings (points, GD, GF tiebreakers) -> top-2 per group plus
the 8 best third-placed teams -> bracket slot resolution -> knockout
propagation. Vectorized over N simulations. Returns, for every knockout slot,
the modal occupants (the predicted matchup) and champion probabilities.

The per-match SCORE / winner / penalties predictions are computed deterministically
elsewhere (run.py) from the model on the modal matchup, so they stay internally
consistent; the simulator's job is to work out *who plays whom*.
"""
import re
import numpy as np
from . import config


def _match_mu(model, eff_h, eff_a, host_h):
    gap = eff_h - eff_a
    mu_h = np.exp(np.clip(model.beta[0] + model.beta[1] * gap / 100.0
                          + model.beta[2] * host_h, -10, 10))
    mu_a = np.exp(np.clip(model.beta[0] - model.beta[1] * gap / 100.0, -10, 10))
    return mu_h, mu_a


def _bipartite_match(groups, slots_allowed):
    """Perfect matching group->slot where group in slot's allowed set (Kuhn's)."""
    slot_to_group = {s: None for s in slots_allowed}
    def try_assign(g, seen):
        for s, allowed in slots_allowed.items():
            if g in allowed and s not in seen:
                seen.add(s)
                if slot_to_group[s] is None or try_assign(slot_to_group[s], seen):
                    slot_to_group[s] = g
                    return True
        return False
    for g in groups:
        try_assign(g, set())
    return slot_to_group  # {slot: group}


def run(gf, ks, disp_of_cell, idx_of_disp, eff_by_idx, is_host_by_idx, model,
        n=None, seed=None):
    n = n or config.N_SIMS
    rng = np.random.default_rng(seed or config.RANDOM_SEED)

    groups = sorted(gf["group"].unique())
    # team indices per group
    g_team_idx = {}
    for g in groups:
        sub = gf[gf["group"] == g]
        cells = sorted(set(sub["home_team"]) | set(sub["away_team"]))
        g_team_idx[g] = [idx_of_disp[disp_of_cell(c)] for c in cells]

    # accumulators per group: pts/gd/gf shape (4, n)
    g_stats = {g: np.zeros((len(g_team_idx[g]), 3, n)) for g in groups}
    for g in groups:
        local = {t: k for k, t in enumerate(g_team_idx[g])}
        sub = gf[gf["group"] == g]
        for r in sub.itertuples(index=False):
            hi = idx_of_disp[disp_of_cell(r.home_team)]
            ai = idx_of_disp[disp_of_cell(r.away_team)]
            host_h = 1.0 if is_host_by_idx[hi] else 0.0
            mu_h, mu_a = _match_mu(model, eff_by_idx[hi], eff_by_idx[ai], host_h)
            hg = rng.poisson(mu_h, n); ag = rng.poisson(mu_a, n)
            hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            lh, la = local[hi], local[ai]
            g_stats[g][lh, 0] += hp; g_stats[g][la, 0] += ap
            g_stats[g][lh, 1] += hg - ag; g_stats[g][la, 1] += ag - hg
            g_stats[g][lh, 2] += hg; g_stats[g][la, 2] += ag

    # rank teams within each group (pts, gd, gf, random tiebreak)
    win_idx, run_idx, third_idx, third_stats = {}, {}, {}, {}
    for g in groups:
        st = g_stats[g]
        noise = rng.random((len(g_team_idx[g]), n)) * 1e-3
        score = st[:, 0] * 1e6 + st[:, 1] * 1e3 + st[:, 2] + noise
        order = np.argsort(-score, axis=0)             # (4, n)
        teams_arr = np.array(g_team_idx[g])
        win_idx[g] = teams_arr[order[0]]
        run_idx[g] = teams_arr[order[1]]
        third_idx[g] = teams_arr[order[2]]
        # third place stats for best-3rd ranking
        rows = order[2]
        third_stats[g] = np.stack([st[rows, 0, np.arange(n)],
                                   st[rows, 1, np.arange(n)],
                                   st[rows, 2, np.arange(n)]], axis=0)  # (3,n)

    # best 8 thirds and slot assignment (cached by qualifying-group set)
    third_score = np.stack([third_stats[g][0] * 1e6 + third_stats[g][1] * 1e3
                            + third_stats[g][2] + rng.random(n) * 1e-3
                            for g in groups], axis=0)        # (12, n)
    best8_groups = np.argsort(-third_score, axis=0)[:8]      # (8, n) indices into `groups`

    # parse best-3rd slot allowed sets
    best3_slots = {}
    for s in set(ks["slot_home"]) | set(ks["slot_away"]):
        m = re.match(r"Best 3rd \(Groups ([A-L/]+)\)", str(s))
        if m:
            best3_slots[s] = set(m.group(1).split("/"))

    slot_team = {s: np.full(n, -1) for s in best3_slots}      # best-3rd slot -> team idx per sim
    cache = {}
    for sim in range(n):
        qual = frozenset(groups[best8_groups[k, sim]] for k in range(8))
        assign = cache.get(qual)
        if assign is None:
            assign = _bipartite_match(list(qual), best3_slots)
            cache[qual] = assign
        for s, g in assign.items():
            if g is not None:
                slot_team[s][sim] = third_idx[g][sim]

    # ---- resolve every slot string to a team-idx array ----
    ko = {}  # match_id -> dict(home, away, winner, loser, pens)

    def resolve(slot):
        slot = str(slot)
        m = re.match(r"Winner Group ([A-L])", slot)
        if m: return win_idx[m.group(1)]
        m = re.match(r"Runner-up Group ([A-L])", slot)
        if m: return run_idx[m.group(1)]
        if slot in slot_team: return slot_team[slot]
        m = re.match(r"Winner Match (\d+)", slot)
        if m: return ko[int(m.group(1))]["winner"]
        m = re.match(r"Loser Match (\d+)", slot)
        if m: return ko[int(m.group(1))]["loser"]
        raise ValueError(f"Unresolved slot: {slot}")

    ks_sorted = ks.sort_values("match_id")
    for r in ks_sorted.itertuples(index=False):
        hi = resolve(r.slot_home); ai = resolve(r.slot_away)
        eff_h = eff_by_idx[hi]; eff_a = eff_by_idx[ai]
        gap = eff_h - eff_a
        mu_h = np.exp(np.clip(model.beta[0] + model.beta[1] * gap / 100.0, -10, 10))
        mu_a = np.exp(np.clip(model.beta[0] - model.beta[1] * gap / 100.0, -10, 10))
        hg = rng.poisson(mu_h); ag = rng.poisson(mu_a)
        drawn = hg == ag
        # winner: regulation result, else Elo-style coin weighted by rating
        p_home = 1.0 / (1.0 + 10 ** ((eff_a - eff_h) / 400.0))
        coin = rng.random(n) < p_home
        winner = np.where(hg > ag, hi, np.where(ag > hg, ai, np.where(coin, hi, ai)))
        loser = np.where(winner == hi, ai, hi)
        # penalties: drawn after 90 -> ~50% reach a shootout (rest settled in ET)
        pens = drawn & (rng.random(n) < 0.5)
        ko[int(r.match_id)] = dict(home=hi, away=ai, winner=winner,
                                   loser=loser, pens=pens)

    # ---- aggregate modal occupants per knockout match ----
    def mode_team(arr):
        return int(np.bincount(arr, minlength=len(eff_by_idx)).argmax())

    ko_out = {}
    for mid, d in ko.items():
        ko_out[mid] = dict(
            home_idx=mode_team(d["home"]),
            away_idx=mode_team(d["away"]),
            p_pens=float(d["pens"].mean()),
        )

    final_mid = int(ks[ks["round"] == "Final"]["match_id"].iloc[0])
    champ = ko[final_mid]["winner"]
    champ_prob = {int(t): float((champ == t).mean()) for t in np.unique(champ)}

    return dict(ko=ko_out, champion_prob=champ_prob,
                win_idx=win_idx, run_idx=run_idx)
