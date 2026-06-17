"""Orchestrator: load data -> Elo -> fit Poisson -> effective ratings ->
group predictions (deterministic) + knockout predictions (Monte-Carlo bracket).

Public entry points:
    prepare()            -> context dict with model, ratings, sim results
    predict_group(ctx)   -> DataFrame in the group_predictions schema
    predict_knockout(ctx)-> DataFrame in the knockout_predictions schema
    main()               -> runs everything, writes predictions/*.csv
    backtest(year)       -> leak-free quality check on a past World Cup
"""
import numpy as np
import pandas as pd
from . import config, teams, elo as elo_mod, features as feat_mod, model as model_mod, scoring, simulate


def prepare(verbose=True):
    if not config.RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {config.RESULTS_CSV}. Download martj42 results.csv into the data/ folder.")
    gf = pd.read_csv(config.GROUP_FIXTURES)
    ks = pd.read_csv(config.KNOCKOUT_SLOTS)
    results = feat_mod.load_results()
    feat_df = teams.load_team_features()
    placeholders = teams.load_placeholders()
    available = set(results["home_team"]) | set(results["away_team"])

    # cells -> display names -> results names -> integer ids
    cells = sorted(set(gf["home_team"]) | set(gf["away_team"]))
    disp_of_cell = {c: teams.resolve_fixture_team(c, placeholders) for c in cells}
    displays = sorted(set(disp_of_cell.values()))
    idx_of_disp = {d: i for i, d in enumerate(displays)}
    disp_to_results, unresolved = {}, []
    for d in displays:
        rn = teams.resolve_to_results(d, available)
        if rn is None:
            unresolved.append(d)
        disp_to_results[d] = rn
    if unresolved and verbose:
        print("WARNING unresolved team names (fallback to average strength):", unresolved)

    # Elo from full history; goals model on recent window
    latest_elo, train_all = elo_mod.compute(results)
    cutoff = train_all["date"].max() - pd.Timedelta(days=365 * config.TRAIN_YEARS)
    train = train_all[train_all["date"] >= cutoff].reset_index(drop=True)
    X, y, w = model_mod.build_training(train)
    model = model_mod.PoissonGLM().fit(X, y, w)
    rho = model_mod.fit_rho(train, model)

    # effective ratings, keyed by results name then mapped to idx
    wc_keys = {disp_to_results[d]: teams.norm(d) for d in displays if disp_to_results[d]}
    eff_by_results = feat_mod.build_effective_ratings(latest_elo, feat_df, wc_keys)
    eff_by_idx = np.full(len(displays), config.ELO_START)
    is_host_by_idx = np.zeros(len(displays), bool)
    for d, i in idx_of_disp.items():
        rn = disp_to_results[d]
        if rn:
            eff_by_idx[i] = eff_by_results.get(rn, latest_elo.get(rn, config.ELO_START))
            is_host_by_idx[i] = rn in config.HOST_NATIONS

    sim = simulate.run(gf, ks, lambda c: disp_of_cell[c], idx_of_disp,
                       eff_by_idx, is_host_by_idx, model)

    if verbose:
        print(f"Fitted weights  intercept={model.beta[0]:.3f}  "
              f"rating(/100)={model.beta[1]:.3f}  home={model.beta[2]:.3f}  rho={rho:.3f}")
        top = sorted(sim["champion_prob"].items(), key=lambda kv: -kv[1])[:8]
        disp_of_idx = {i: d for d, i in idx_of_disp.items()}
        print("Title odds:", ", ".join(f"{disp_of_idx[i]} {p*100:.1f}%" for i, p in top))

    return dict(gf=gf, ks=ks, model=model, rho=rho, eff_by_idx=eff_by_idx,
                is_host_by_idx=is_host_by_idx, idx_of_disp=idx_of_disp,
                disp_of_idx={i: d for d, i in idx_of_disp.items()},
                disp_of_cell=disp_of_cell, sim=sim)


def _predict_match(ctx, hi, ai, host_h):
    m = ctx["model"]; gap = ctx["eff_by_idx"][hi] - ctx["eff_by_idx"][ai]
    mu_h = float(np.exp(np.clip(m.beta[0] + m.beta[1]*gap/100 + m.beta[2]*host_h, -10, 10)))
    mu_a = float(np.exp(np.clip(m.beta[0] - m.beta[1]*gap/100, -10, 10)))
    M = model_mod.score_matrix(mu_h, mu_a, ctx["rho"])
    (pi, pj), _ = scoring.optimal_scoreline(M)
    ph, pd_, pa, egh, ega = model_mod.outcome_probs(M)
    return pi, pj, ph, pd_, pa


def predict_group(ctx):
    gf = ctx["gf"].copy()
    rows = []
    for r in gf.itertuples(index=False):
        hi = ctx["idx_of_disp"][ctx["disp_of_cell"][r.home_team]]
        ai = ctx["idx_of_disp"][ctx["disp_of_cell"][r.away_team]]
        host_h = 1.0 if ctx["is_host_by_idx"][hi] else 0.0
        pi, pj, ph, pd_, pa = _predict_match(ctx, hi, ai, host_h)
        rows.append((pi, pj, config.CORNERS_GROUP, config.YELLOWS_GROUP, config.REDS,
                     scoring.outcome_from_probs(ph, pd_, pa)))
    out = pd.DataFrame(rows, columns=["predicted_home_goals", "predicted_away_goals",
                                      "corners", "yellow_cards", "red_cards", "winning_team"])
    return pd.concat([gf.reset_index(drop=True), out], axis=1)


def predict_knockout(ctx):
    ks = ctx["ks"].copy()
    di = ctx["disp_of_idx"]
    rows = []
    for r in ks.itertuples(index=False):
        d = ctx["sim"]["ko"][int(r.match_id)]
        hi, ai = d["home_idx"], d["away_idx"]
        pi, pj, ph, pd_, pa = _predict_match(ctx, hi, ai, 0.0)
        penalties = bool((pd_ * 0.5) > 0.5)   # shootout only if draw is very likely
        rows.append((di[hi], di[ai], pi, pj, config.CORNERS_KO, config.YELLOWS_KO,
                     config.REDS, scoring.ko_winner(ph, pa), penalties))
    out = pd.DataFrame(rows, columns=["predicted_home_team", "predicted_away_team",
                                      "predicted_home_goals", "predicted_away_goals",
                                      "corners", "yellow_cards", "red_cards",
                                      "match_winner", "penalties"])
    return pd.concat([ks.reset_index(drop=True), out], axis=1)


def backtest(year=2022, verbose=True):
    """Leak-free check: predict every World Cup match in `year` using only Elo
    available before each match, and score against actuals with the rubric."""
    results = feat_mod.load_results()
    _, train_all = elo_mod.compute(results)
    wc = train_all[(train_all["tournament"].str.contains("FIFA World Cup", case=False))
                   & (train_all["date"].dt.year == year)]
    cut = train_all["date"].max() - pd.Timedelta(days=365*config.TRAIN_YEARS)
    tr = train_all[(train_all["date"] >= cut) & (train_all["date"] < wc["date"].min())]
    X, y, w = model_mod.build_training(tr)
    m = model_mod.PoissonGLM().fit(X, y, w)
    rho = model_mod.fit_rho(tr, m)
    tot, exact, winr, n = 0, 0, 0, 0
    for r in wc.itertuples(index=False):
        gap = r.elo_home_pre - r.elo_away_pre
        hf = 0.0 if r.neutral else 1.0
        mu_h = m.mu(gap, hf); mu_a = m.mu(-gap, 0.0)
        M = model_mod.score_matrix(mu_h, mu_a, rho)
        (pi, pj), _ = scoring.optimal_scoreline(M)
        ph, pd_, pa, _, _ = model_mod.outcome_probs(M)
        pts = scoring.score_points(pi, pj, r.home_score, r.away_score)
        if pi == r.home_score and pj == r.away_score:
            exact += 1
        actual = "home" if r.home_score > r.away_score else ("away" if r.home_score < r.away_score else "draw")
        if scoring.outcome_from_probs(ph, pd_, pa) == actual:
            winr += 1; pts += 40
        tot += pts; n += 1
    if verbose:
        print(f"Backtest WC{year}: {n} matches | avg pts/match {tot/n:.1f} | "
              f"exact score {exact}/{n} ({exact/n*100:.0f}%) | outcome {winr}/{n} ({winr/n*100:.0f}%)")
    return dict(n=n, total=tot, exact=exact, outcome=winr)


def main():
    ctx = prepare()
    g = predict_group(ctx); k = predict_knockout(ctx)
    config.OUT.mkdir(exist_ok=True)
    g.to_csv(config.OUT / "group_predictions.csv", index=False)
    k.to_csv(config.OUT / "knockout_predictions.csv", index=False)
    print(f"Wrote {config.OUT/'group_predictions.csv'} and knockout_predictions.csv")
    return ctx, g, k


if __name__ == "__main__":
    main()
