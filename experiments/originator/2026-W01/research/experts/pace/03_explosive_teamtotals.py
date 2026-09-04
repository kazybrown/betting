"""03: THEORY 2 -- explosive offense vs weak explosive defense and TEAM scoring beyond the
market-implied team total (home = T/2 - S/2, away = T/2 + S/2; S in ORIGINATOR convention,
negative = home favored). Long format: two rows per game (team, opp). SEs clustered by game.
Explosive play = pass >= 20 yds or rush >= 10 yds (rate per offensive play); expl20 = any play
>= 20 yds. Prior-only features (league-relative): off_expl (team offense), opp_def_expl
(what the opponent's defense allowed). Controls: prior EPA/play for / against, points for / against.
(a) market residual: (team_pts - implied_tt) ~ features; OOS MAE of implied_tt + train-fit adj.
(b) market-free baseline for team points (fit train): team_pts - lg_blend/2 ~ team Elo, opp Elo,
    team pf, opp pa, team/opp QB adj, home, dome, wind, div  -> then + explosive features.
(c) matchup rule: explosive offense (top quartile) vs weak explosive defense (top quartile allowed).
Fit 2009-2019, test 2023-2025 REG; rolling-origin check inside 2013-2019.
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experts" / "totals"))
from common import mae, paired_mae_ci, ou_rate
pd.set_option("display.width", 220)

m = pd.read_csv(HERE / "_game_features.csv", low_memory=False)
m["dome"] = m.is_dome.astype(int)
m = m[(m.game_type == "REG") & m.mkt_total.notna() & m.mkt_spread.notna()].copy()

TF = ["expl_off", "expl20_off", "expl_def", "expl20_def", "epa_off", "epa_def", "pts", "pts_allowed", "ppd", "succ_off", "succ_def", "plays", "spp_neut", "proe", "pr_neut"]


def long(m, v):
    rows = []
    for side, opp in (("h", "a"), ("a", "h")):
        x = pd.DataFrame({"gid": m.gid, "season": m.season, "week": m.week, "train": m.train, "test": m.test,
                          "team_pts": m.home_score if side == "h" else m.away_score,
                          "implied_tt": m.implied_home_tt if side == "h" else m.implied_away_tt,
                          "is_home": 1 if side == "h" else 0, "lg_half": m.lg_blend / 2,
                          "team_elo": m.home_pts_vs_avg if side == "h" else m.away_pts_vs_avg,
                          "opp_elo": m.away_pts_vs_avg if side == "h" else m.home_pts_vs_avg,
                          "team_pf": m.h_pf if side == "h" else m.a_pf, "opp_pa": m.a_pa if side == "h" else m.h_pa,
                          "team_qb": (m.home_538_qb_adj if side == "h" else m.away_538_qb_adj).fillna(0) / 25,
                          "opp_qb": (m.away_538_qb_adj if side == "h" else m.home_538_qb_adj).fillna(0) / 25,
                          "dome": m.dome, "wind_f": m.wind_f, "div": m["div"], "mkt_total": m.mkt_total, "total_pts": m.total_pts})
        for f in TF:
            x["off_" + f] = m[f"{side}_{f}_{v}"]
            x["opp_" + f] = m[f"{opp}_{f}_{v}"]
        rows.append(x)
    d = pd.concat(rows, ignore_index=True)
    d["y_mkt"] = d.team_pts - d.implied_tt
    d["mismatch"] = d.off_expl_off * d.opp_expl_def * 100      # (+) explosive O vs leaky D; scaled: rates in 0.01 units
    d["mismatch20"] = d.off_expl20_off * d.opp_expl20_def * 100
    return d


BASE = ["team_elo", "opp_elo", "team_pf", "opp_pa", "team_qb", "opp_qb", "is_home", "dome", "wind_f", "div"]


def ols(y, X, groups):
    X = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})


def fit_pred(tr, te, cols, y="team_pts", offset="lg_half"):
    f = ols(tr[y] - tr[offset], tr[cols], tr.gid)
    return f, np.asarray(te[offset] + f.predict(sm.add_constant(te[cols].astype(float), has_constant="add")))


for v, vlab in [("r8d", "prior-8-game rolling (league-relative)"), ("bld", "prior-season/season-to-date blend K=4")]:
    print("\n" + "=" * 110 + f"\nFEATURE VERSION: {vlab}\n" + "=" * 110)
    d = long(m, v)
    need = BASE + ["off_expl_off", "opp_expl_def", "off_epa_off", "opp_epa_def", "lg_half", "implied_tt"]
    d = d[d[need].notna().all(axis=1)].copy()
    tr, te = d[d.train], d[d.test]
    print(f"sample (team-games): train n={len(tr)} ({tr.gid.nunique()} games) | test n={len(te)} ({te.gid.nunique()} games)")
    print(f"sanity: corr(implied_tt, team_pts) train {stats.pearsonr(tr.implied_tt, tr.team_pts)[0]:+.3f} test {stats.pearsonr(te.implied_tt, te.team_pts)[0]:+.3f}; "
          f"MAE implied_tt: train {mae(tr.implied_tt, tr.team_pts):.3f} test {mae(te.implied_tt, te.team_pts):.3f}")
    print(f"descriptive: corr(off_expl prior, team_pts) train {stats.pearsonr(tr.off_expl_off, tr.team_pts)[0]:+.3f} | corr(opp_def_expl prior, team_pts) {stats.pearsonr(tr.opp_expl_def, tr.team_pts)[0]:+.3f} "
          f"| corr(off_expl prior, implied_tt) {stats.pearsonr(tr.off_expl_off, tr.implied_tt)[0]:+.3f} (market prices it?)")

    print("\n-- (a) MARKET residual: (team_pts - implied_tt) ~ x  [cluster-by-game SE] --")
    print(f"  {'x':34s} {'train coef (se) p':>26s} {'test coef (se) p':>26s}   OOS dMAE(implied+adj) vs implied [95% CI]")
    specs = [("off_expl_off", ["off_expl_off"]), ("opp_expl_def", ["opp_expl_def"]), ("both additive", ["off_expl_off", "opp_expl_def"]),
             ("mismatch (product)", ["mismatch"]), ("additive + mismatch", ["off_expl_off", "opp_expl_def", "mismatch"]),
             ("expl20 additive", ["off_expl20_off", "opp_expl20_def"]), ("expl20 mismatch", ["mismatch20"]),
             ("additive + EPA controls", ["off_expl_off", "opp_expl_def", "off_epa_off", "opp_epa_def"]),
             ("EPA only", ["off_epa_off", "opp_epa_def"]), ("pts for/against only", ["off_pts", "opp_pts_allowed"])]
    for lab, cols in specs:
        f1 = ols(tr.y_mkt, tr[cols], tr.gid); f2 = ols(te.y_mkt, te[cols], te.gid)
        adj = f1.predict(sm.add_constant(te[cols].astype(float), has_constant="add"))
        dm, lo, hi, n = paired_mae_ci(te.team_pts - (te.implied_tt + adj), te.y_mkt)
        c = cols[0]
        print(f"  {lab:34s} {f1.params[c]:+8.3f} ({f1.bse[c]:.3f}) p={f1.pvalues[c]:.3f} {f2.params[c]:+8.3f} ({f2.bse[c]:.3f}) p={f2.pvalues[c]:.3f}   {dm:+.3f} [{lo:+.3f},{hi:+.3f}]"
              + ("" if len(cols) == 1 else "   | other coefs: " + ", ".join(f"{k}={f1.params[k]:+.3f}(p={f1.pvalues[k]:.2f})" for k in cols[1:])))

    print("\n-- (b) MARKET-FREE team-points baseline + explosive features (fit train, OOS 2023-25) --")
    fb, pb = fit_pred(tr, te, BASE)
    dm, lo, hi, n = paired_mae_ci(te.team_pts - pb, te.y_mkt)
    print(f"  BASE: test MAE={mae(pb, te.team_pts):.3f} vs implied_tt {mae(te.implied_tt, te.team_pts):.3f} (dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}]); "
          f"coefs " + ", ".join(f"{k}={fb.params[k]:+.2f}" for k in BASE))
    sd_o, sd_d = tr.off_expl_off.std(), tr.opp_expl_def.std()
    print(f"  1 SD of prior explosive rate: offense {sd_o:.4f}, defense-allowed {sd_d:.4f} (rate per play; league mean ~0.098)")
    for lab, cols in [("+ off_expl", ["off_expl_off"]), ("+ opp_def_expl", ["opp_expl_def"]), ("+ both additive", ["off_expl_off", "opp_expl_def"]),
                      ("+ mismatch", ["mismatch"]), ("+ additive + mismatch", ["off_expl_off", "opp_expl_def", "mismatch"]),
                      ("+ expl20 additive", ["off_expl20_off", "opp_expl20_def"]),
                      ("+ EPA for/against", ["off_epa_off", "opp_epa_def"]), ("+ EPA + explosive", ["off_epa_off", "opp_epa_def", "off_expl_off", "opp_expl_def"]),
                      ("+ ppd + explosive", ["off_ppd", "off_expl_off", "opp_expl_def"])]:
        f, p = fit_pred(tr, te, BASE + cols)
        dm, lo, hi, n = paired_mae_ci(te.team_pts - p, te.team_pts - pb)
        dmm, lom, him, _ = paired_mae_ci(te.team_pts - p, te.y_mkt)
        print(f"  {lab:24s} test MAE={mae(p, te.team_pts):.3f}  dMAE vs BASE {dm:+.3f} [{lo:+.3f},{hi:+.3f}]  vs implied {dmm:+.3f} [{lom:+.3f},{him:+.3f}]  | "
              + ", ".join(f"{k}={f.params[k]:+.3f}(se {f.bse[k]:.3f},p={f.pvalues[k]:.2f}; per SD {f.params[k]*tr[k].std():+.2f})" for k in cols))

    print("\n-- (b2) rolling-origin (fit < Y, test Y): BASE vs BASE+explosive additive, team-points MAE --")
    res = []
    for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
        trY = d[(d.season < Y)]; teY = d[d.season == Y]
        f0, p0 = fit_pred(trY, teY, BASE); f1, p1 = fit_pred(trY, teY, BASE + ["off_expl_off", "opp_expl_def"])
        res.append((Y, len(teY), mae(p0, teY.team_pts), mae(p1, teY.team_pts), mae(teY.implied_tt, teY.team_pts), f1.params["off_expl_off"], f1.params["opp_expl_def"]))
    r = pd.DataFrame(res, columns=["Y", "n", "BASE", "BASE+expl", "implied_tt", "coef_off", "coef_def"])
    print(r.round(3).to_string(index=False)); print(f"  mean BASE={r.BASE.mean():.3f} BASE+expl={r['BASE+expl'].mean():.3f} implied={r.implied_tt.mean():.3f}")

    print("\n-- (c) matchup rule: explosive offense (top quartile prior) vs weak explosive defense (top-quartile allowed); cutoffs from train --")
    qo = np.percentile(tr.off_expl_off, [25, 75]); qd = np.percentile(tr.opp_expl_def, [25, 75])
    d["ocls"] = np.where(d.off_expl_off >= qo[1], "X", np.where(d.off_expl_off <= qo[0], "x", "-"))
    d["dcls"] = np.where(d.opp_expl_def >= qd[1], "W", np.where(d.opp_expl_def <= qd[0], "s", "-"))   # W = weak (allows many), s = stingy
    d["cell"] = d.ocls + d.dcls
    fb2, pall = fit_pred(d[d.train], d, BASE); d["base_res"] = d.team_pts - pall
    for lab, sub in [("TRAIN (baseline in-sample)", d[d.train]), ("TEST 2023-25", d[d.test])]:
        gb = sub.groupby("cell").agg(n=("gid", "size"), pts=("team_pts", "mean"), implied=("implied_tt", "mean"), res_mkt=("y_mkt", "mean"), sd_m=("y_mkt", "std"),
                                     res_base=("base_res", "mean"), sd_b=("base_res", "std"))
        gb["se_mkt"] = gb.sd_m / np.sqrt(gb.n); gb["se_base"] = gb.sd_b / np.sqrt(gb.n)
        print(f"  {lab}: cells (offense X=explosive/x=not, defense W=weak/s=stingy)")
        print(gb[["n", "pts", "implied", "res_mkt", "se_mkt", "res_base", "se_base"]].round(2).loc[[c for c in ["XW", "X-", "Xs", "-W", "--", "-s", "xW", "x-", "xs"] if c in gb.index]].to_string())
    for lab, sub in [("TRAIN", d[d.train]), ("TEST", d[d.test])]:
        a = sub[sub.cell == "XW"]; b = sub[sub.cell == "xs"]
        t1, p1 = stats.ttest_ind(a.base_res, b.base_res, equal_var=False); t2, p2 = stats.ttest_ind(a.y_mkt, b.y_mkt, equal_var=False)
        print(f"  {lab} XW vs xs: market-free residual diff={a.base_res.mean()-b.base_res.mean():+.2f} (p={p1:.3f}); market residual diff={a.y_mkt.mean()-b.y_mkt.mean():+.2f} (p={p2:.3f}); n={len(a)}/{len(b)}")
