"""04: THEORY 3 -- pass rate over expected (PROE) as a TOTALS signal.
PROE = pass share minus expected pass share from a logit fit on 2009-2019 pbp (down, distance,
field position, score x time, clock, win probability); league-relative, prior-only versions
(rolling 8 / blend). Also the raw neutral pass rate and a 'style shift' = season-to-date PROE
minus prior-season PROE (teams that changed play-calling).
Game level: total ~ proe_sum; team level: team_pts ~ own PROE, opp PROE (cluster SE by game).
Both vs the MARKET (residual regression, OOS MAE) and vs the MARKET-FREE LEAN baseline.
Fit 2009-2019, test 2023-2025 REG; rolling-origin inside 2013-2019 as extra evidence.
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
tgf = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
# style shift: season-to-date PROE (before this game, >= 4 games) minus prior-season PROE
tgf["proe_shift"] = np.where(tgf.gp >= 4, tgf.proe_ytd - tgf.proe_prev, np.nan)
tgf["pr_shift"] = np.where(tgf.gp >= 4, tgf.pr_neut_ytd - tgf.pr_neut_prev, np.nan)
sh = tgf[["gid", "team", "proe_shift", "pr_shift", "proe_ytd", "proe_prev"]]
m = m.merge(sh.rename(columns={"team": "home", "proe_shift": "h_proe_shift", "pr_shift": "h_pr_shift", "proe_ytd": "h_proe_ytd", "proe_prev": "h_proe_prev"}), on=["gid", "home"], how="left")
m = m.merge(sh.rename(columns={"team": "away", "proe_shift": "a_proe_shift", "pr_shift": "a_pr_shift", "proe_ytd": "a_proe_ytd", "proe_prev": "a_proe_prev"}), on=["gid", "away"], how="left")
BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "div"]


def ols(y, X, groups=None):
    X = sm.add_constant(X.astype(float), has_constant="add")
    if groups is None:
        return sm.OLS(np.asarray(y, float), X).fit(cov_type="HC1")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})


def fit_pred(tr, te, cols, y="total_pts", offset="lg_blend", groups=None):
    f = ols(tr[y] - tr[offset], tr[cols], None if groups is None else tr[groups])
    return f, np.asarray(te[offset] + f.predict(sm.add_constant(te[cols].astype(float), has_constant="add")))


for v, vlab in [("r8d", "prior-8-game rolling (league-relative)"), ("bld", "prior-season/season-to-date blend K=4")]:
    print("\n" + "=" * 110 + f"\nFEATURE VERSION: {vlab}\n" + "=" * 110)
    d = m.copy()
    d["proe_sum"] = d[f"h_proe_{v}"] + d[f"a_proe_{v}"]
    d["proe_neut_sum"] = d[f"h_proe_neut_{v}"] + d[f"a_proe_neut_{v}"]
    d["pr_sum"] = d[f"h_pr_neut_{v}"] + d[f"a_pr_neut_{v}"]
    d["pr_all_sum"] = d[f"h_pr_all_{v}"] + d[f"a_pr_all_{v}"]
    d["pr_wp_sum"] = d[f"h_pr_wp_{v}"] + d[f"a_pr_wp_{v}"]
    d["proe_max"] = d[[f"h_proe_{v}", f"a_proe_{v}"]].max(axis=1)
    d["proe_absdiff"] = (d[f"h_proe_{v}"] - d[f"a_proe_{v}"]).abs()
    d["spp_avg"] = (d[f"h_spp_neut_{v}"] + d[f"a_spp_neut_{v}"]) / 2
    d["epa_sum"] = d[f"h_epa_off_{v}"] + d[f"a_epa_off_{v}"] + d[f"h_epa_def_{v}"] + d[f"a_epa_def_{v}"]
    d["shift_sum"] = d.h_proe_shift + d.a_proe_shift
    d = d[(d.game_type == "REG") & d.mkt_total.notna() & d[BASE + ["lg_blend", "proe_sum", "pr_sum", "spp_avg", "epa_sum"]].notna().all(axis=1)].copy()
    tr, te = d[d.train], d[d.test]
    print(f"sample: train n={len(tr)} | test n={len(te)}  | 1 SD proe_sum (train) = {tr.proe_sum.std():.2f} pct-pts, pr_sum = {tr.pr_sum.std():.3f}")
    print("descriptive corr with total_pts (train | test), with MARKET total, with realized game plays:")
    for c in ["proe_sum", "proe_neut_sum", "pr_sum", "pr_all_sum", "pr_wp_sum", "proe_max", "proe_absdiff"]:
        gp_tr = tr.h_plays_act + tr.a_plays_act; gp_te = te.h_plays_act + te.a_plays_act
        print(f"  {c:14s} total {stats.pearsonr(tr[c], tr.total_pts)[0]:+.3f} | {stats.pearsonr(te[c], te.total_pts)[0]:+.3f}   market {stats.pearsonr(tr[c], tr.mkt_total)[0]:+.3f} | {stats.pearsonr(te[c], te.mkt_total)[0]:+.3f}   "
              f"plays {stats.pearsonr(tr[c], gp_tr)[0]:+.3f} | {stats.pearsonr(te[c], gp_te)[0]:+.3f}")

    print("\n-- (a) MARKET residual: (total - mkt_total) ~ x (HC1); OOS = market + train-fit adj --")
    for c in ["proe_sum", "proe_neut_sum", "pr_sum", "pr_all_sum", "pr_wp_sum", "proe_max", "proe_absdiff"]:
        f1 = ols(tr.total_err_mkt, tr[[c]]); f2 = ols(te.total_err_mkt, te[[c]])
        adj = f1.params["const"] + f1.params[c] * te[c]
        dm, lo, hi, n = paired_mae_ci(te.total_pts - (te.mkt_total + adj), te.total_err_mkt)
        w, l, p = ou_rate(te.mkt_total + adj, te.mkt_total, te.total_pts)
        print(f"  {c:14s} train {f1.params[c]:+7.3f} (se {f1.bse[c]:.3f}) p={f1.pvalues[c]:.3f} [per SD {f1.params[c]*tr[c].std():+.2f}] | test {f2.params[c]:+7.3f} (se {f2.bse[c]:.3f}) p={f2.pvalues[c]:.3f} | "
              f"OOS dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}] O/U {w}-{l}-{p} ({w/max(w+l,1):.3f})")
    s = d.dropna(subset=["shift_sum"]); str_, ste = s[s.train], s[s.test]
    f1 = ols(str_.total_err_mkt, str_[["shift_sum"]]); f2 = ols(ste.total_err_mkt, ste[["shift_sum"]])
    print(f"  style shift (ytd PROE - prev-season PROE, sum): train n={len(str_)} coef {f1.params['shift_sum']:+.3f} (se {f1.bse['shift_sum']:.3f}) p={f1.pvalues['shift_sum']:.3f} | "
          f"test n={len(ste)} coef {f2.params['shift_sum']:+.3f} (se {f2.bse['shift_sum']:.3f}) p={f2.pvalues['shift_sum']:.3f}")

    print("\n-- (b) MARKET-FREE LEAN baseline + PROE / pass-rate (fit train, OOS 2023-25) --")
    fb, pb = fit_pred(tr, te, BASE)
    print(f"  BASE: test MAE={mae(pb, te.total_pts):.3f}; market {mae(te.mkt_total, te.total_pts):.3f}")
    for lab, cols in [("+ proe_sum", ["proe_sum"]), ("+ proe_neut_sum", ["proe_neut_sum"]), ("+ pr_sum (neutral)", ["pr_sum"]), ("+ pr_all_sum", ["pr_all_sum"]),
                      ("+ proe_sum + spp_avg", ["proe_sum", "spp_avg"]), ("+ pr_sum + spp_avg", ["pr_sum", "spp_avg"]),
                      ("+ proe_sum + epa_sum", ["proe_sum", "epa_sum"]), ("+ epa_sum only", ["epa_sum"]),
                      ("+ proe_max", ["proe_max"]), ("+ h/a proe separately", [f"h_proe_{v}", f"a_proe_{v}"])]:
        f, p = fit_pred(tr, te, BASE + cols)
        dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
        dmm, lom, him, _ = paired_mae_ci(te.total_pts - p, te.total_err_mkt)
        w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
        print(f"  {lab:24s} MAE={mae(p, te.total_pts):.3f} dMAE vs BASE {dm:+.3f} [{lo:+.3f},{hi:+.3f}] vs mkt {dmm:+.3f} [{lom:+.3f},{him:+.3f}] O/U {w}-{l}-{pu} ({w/max(w+l,1):.3f}) | "
              + ", ".join(f"{k}={f.params[k]:+.3f}(se {f.bse[k]:.3f},p={f.pvalues[k]:.2f}; per SD {f.params[k]*tr[k].std():+.2f})" for k in cols))

    print("\n-- (b2) rolling-origin (fit < Y, test Y) BASE vs BASE+proe_sum vs BASE+pr_sum --")
    res = []
    for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
        trY = d[d.season < Y]; teY = d[d.season == Y]
        f0, p0 = fit_pred(trY, teY, BASE); f1, p1 = fit_pred(trY, teY, BASE + ["proe_sum"]); f2, p2 = fit_pred(trY, teY, BASE + ["pr_sum"])
        res.append((Y, len(teY), mae(p0, teY.total_pts), mae(p1, teY.total_pts), mae(p2, teY.total_pts), mae(teY.mkt_total, teY.total_pts), f1.params["proe_sum"], f2.params["pr_sum"]))
    r = pd.DataFrame(res, columns=["Y", "n", "BASE", "+proe", "+pr", "mkt", "coef_proe", "coef_pr"])
    print(r.round(3).to_string(index=False)); print(f"  mean BASE={r.BASE.mean():.3f} +proe={r['+proe'].mean():.3f} +pr={r['+pr'].mean():.3f} mkt={r.mkt.mean():.3f}")

    print("\n-- (c) TEAM level: team_pts - implied_tt ~ own PROE, opp PROE (cluster SE); and market-free --")
    rows = []
    for side, opp in (("h", "a"), ("a", "h")):
        x = pd.DataFrame({"gid": d.gid, "season": d.season, "train": d.train, "test": d.test,
                          "team_pts": d.home_score if side == "h" else d.away_score,
                          "implied_tt": d.implied_home_tt if side == "h" else d.implied_away_tt,
                          "lg_half": d.lg_blend / 2, "is_home": 1 if side == "h" else 0,
                          "team_elo": d.home_pts_vs_avg if side == "h" else d.away_pts_vs_avg, "opp_elo": d.away_pts_vs_avg if side == "h" else d.home_pts_vs_avg,
                          "team_pf": d.h_pf if side == "h" else d.a_pf, "opp_pa": d.a_pa if side == "h" else d.h_pa,
                          "team_qb": (d.home_538_qb_adj if side == "h" else d.away_538_qb_adj).fillna(0) / 25, "opp_qb": (d.away_538_qb_adj if side == "h" else d.home_538_qb_adj).fillna(0) / 25,
                          "dome": d.dome, "wind_f": d.wind_f, "div": d["div"],
                          "own_proe": d[f"{side}_proe_{v}"], "opp_proe": d[f"{opp}_proe_{v}"], "own_pr": d[f"{side}_pr_neut_{v}"], "opp_pr": d[f"{opp}_pr_neut_{v}"],
                          "own_spp": d[f"{side}_spp_neut_{v}"], "opp_spp": d[f"{opp}_spp_neut_{v}"]})
        rows.append(x)
    L = pd.concat(rows, ignore_index=True).dropna(subset=["implied_tt"])
    L["y_mkt"] = L.team_pts - L.implied_tt
    Ltr, Lte = L[L.train], L[L.test]
    TB = ["team_elo", "opp_elo", "team_pf", "opp_pa", "team_qb", "opp_qb", "is_home", "dome", "wind_f", "div"]
    for lab, cols in [("own_proe", ["own_proe"]), ("opp_proe", ["opp_proe"]), ("own+opp proe", ["own_proe", "opp_proe"]), ("own+opp pass rate", ["own_pr", "opp_pr"])]:
        f1 = ols(Ltr.y_mkt, Ltr[cols], Ltr.gid); f2 = ols(Lte.y_mkt, Lte[cols], Lte.gid)
        adj = f1.predict(sm.add_constant(Lte[cols].astype(float), has_constant="add"))
        dm, lo, hi, n = paired_mae_ci(Lte.team_pts - (Lte.implied_tt + adj), Lte.y_mkt)
        print(f"  market residual ~ {lab:18s} train " + ", ".join(f"{k}={f1.params[k]:+.3f}(p={f1.pvalues[k]:.2f})" for k in cols)
              + " | test " + ", ".join(f"{k}={f2.params[k]:+.3f}(p={f2.pvalues[k]:.2f})" for k in cols) + f" | OOS dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}]")
    fb, pb = fit_pred(Ltr, Lte, TB, y="team_pts", offset="lg_half", groups="gid")
    for lab, cols in [("+ own_proe", ["own_proe"]), ("+ own+opp proe", ["own_proe", "opp_proe"]), ("+ own+opp pr", ["own_pr", "opp_pr"]), ("+ own+opp proe + spp", ["own_proe", "opp_proe", "own_spp", "opp_spp"])]:
        f, p = fit_pred(Ltr, Lte, TB + cols, y="team_pts", offset="lg_half", groups="gid")
        dm, lo, hi, n = paired_mae_ci(Lte.team_pts - p, Lte.team_pts - pb)
        print(f"  market-free team pts: BASE MAE={mae(pb, Lte.team_pts):.3f} {lab:22s} MAE={mae(p, Lte.team_pts):.3f} dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}] | "
              + ", ".join(f"{k}={f.params[k]:+.3f}(p={f.pvalues[k]:.2f})" for k in cols))
