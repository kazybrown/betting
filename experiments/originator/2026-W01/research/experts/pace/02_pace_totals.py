"""02: THEORY 1 -- does the PRIOR pace / style of both teams predict the realized total
(a) beyond the MARKET closing total, and (b) beyond a market-free rating baseline (what an
ORIGINATOR should add)?  Fit 2009-2019 (all local pbp seasons <= 2021), test 2023-2025 (REG,
games with a market total and prior-8-game features on both sides). Extra OOS evidence:
rolling-origin inside the training era (fit < Y, test Y, 2013-2019), labelled as such.

Pace features are league-relative (team value minus prior-season league mean) so eras with
different league pace do not confound. Combos:
  spp_avg   mean of both teams' neutral seconds/play (+ = slower)    plays_sum  both teams' off. plays/game
  sppr_avg  same, after-run gaps only (tempo w/o pass-rate effect)    gplays_avg total plays in each team's games
  drives_sum, ppd_sum (points per drive), nh_sum (no-huddle), pr_sum (neutral pass rate), proe_sum
Baseline (market-free) = totals expert's LEAN spec fit on the same rows:
  total - lg_blend ~ elo_sum + pf_sum + pa_sum + qb_sum + dome + wind_f + div
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
BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "div"]


def combos(d, v):
    d = d.copy()
    d["spp_avg"] = (d[f"h_spp_neut_{v}"] + d[f"a_spp_neut_{v}"]) / 2
    d["sppr_avg"] = (d[f"h_spp_neut_run_{v}"] + d[f"a_spp_neut_run_{v}"]) / 2
    d["plays_sum"] = d[f"h_plays_{v}"] + d[f"a_plays_{v}"]
    d["gplays_avg"] = (d[f"h_game_plays_{v}"] + d[f"a_game_plays_{v}"]) / 2
    d["drives_sum"] = d[f"h_drives_{v}"] + d[f"a_drives_{v}"]
    d["ppd_sum"] = d[f"h_ppd_{v}"] + d[f"a_ppd_{v}"]
    d["nh_sum"] = d[f"h_nh_neut_{v}"] + d[f"a_nh_neut_{v}"]
    d["pr_sum"] = d[f"h_pr_neut_{v}"] + d[f"a_pr_neut_{v}"]
    d["proe_sum"] = d[f"h_proe_{v}"] + d[f"a_proe_{v}"]
    d["expl_sum"] = d[f"h_expl_off_{v}"] + d[f"a_expl_off_{v}"] + d[f"h_expl_def_{v}"] + d[f"a_expl_def_{v}"]
    d["epa_sum"] = d[f"h_epa_off_{v}"] + d[f"a_epa_off_{v}"] + d[f"h_epa_def_{v}"] + d[f"a_epa_def_{v}"]
    return d


PACE = ["spp_avg", "sppr_avg", "plays_sum", "gplays_avg", "drives_sum", "nh_sum", "pr_sum", "proe_sum", "ppd_sum", "expl_sum", "epa_sum"]


def ols(y, X, cov="HC1"):
    X = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type=cov)


def fit_pred(tr, te, cols, y="total_pts", offset="lg_blend"):
    f = ols(tr[y] - tr[offset], tr[cols])
    p = te[offset] + f.predict(sm.add_constant(te[cols].astype(float), has_constant="add"))
    return f, np.asarray(p)


for v, vlab in [("r8d", "prior-8-game rolling (league-relative)"), ("bld", "prior-season/season-to-date blend K=4 (league-relative)")]:
    print("\n" + "=" * 110 + f"\nFEATURE VERSION: {vlab}\n" + "=" * 110)
    d = combos(m, v)
    d = d[(d.game_type == "REG") & d.mkt_total.notna() & d[PACE + BASE + ["lg_blend"]].notna().all(axis=1)].copy()
    tr, te = d[d.train], d[d.test]
    print(f"sample: train 2009-2019 n={len(tr)} | test 2023-2025 n={len(te)}")
    gp_act = d.h_plays_act + d.a_plays_act
    print("\n-- (A) descriptive: prior pace -> realized game plays / total (corr, train | test) --")
    for c in PACE:
        r1 = stats.pearsonr(tr[c], tr.h_plays_act + tr.a_plays_act)[0]; r2 = stats.pearsonr(te[c], te.h_plays_act + te.a_plays_act)[0]
        t1 = stats.pearsonr(tr[c], tr.total_pts)[0]; t2 = stats.pearsonr(te[c], te.total_pts)[0]
        k1 = stats.pearsonr(tr[c], tr.mkt_total)[0]; k2 = stats.pearsonr(te[c], te.mkt_total)[0]
        print(f"  {c:11s} vs game plays {r1:+.3f} | {r2:+.3f}   vs total_pts {t1:+.3f} | {t2:+.3f}   vs MARKET total {k1:+.3f} | {k2:+.3f}")

    print("\n-- (B) MARKET residual regressions: (total - mkt_total) ~ x, single features, HC1 --")
    print(f"  {'x':11s} {'train coef (se) p':>28s} {'test coef (se) p':>28s}   OOS: MAE(mkt + train-fit adj) - MAE(mkt) [95% CI]  O/U")
    for c in PACE:
        f1 = ols(tr.total_err_mkt, tr[[c]]); f2 = ols(te.total_err_mkt, te[[c]])
        adj = f1.params["const"] + f1.params[c] * te[c]
        dm, lo, hi, n = paired_mae_ci(te.total_pts - (te.mkt_total + adj), te.total_err_mkt)
        w, l, p = ou_rate(te.mkt_total + adj, te.mkt_total, te.total_pts)
        print(f"  {c:11s} {f1.params[c]:+8.3f} ({f1.bse[c]:.3f}) p={f1.pvalues[c]:.3f}   {f2.params[c]:+8.3f} ({f2.bse[c]:.3f}) p={f2.pvalues[c]:.3f}   "
              f"{dm:+.3f} [{lo:+.3f},{hi:+.3f}]  {w}-{l}-{p} ({w/max(w+l,1):.3f})")
    joint = ["spp_avg", "ppd_sum", "pr_sum", "proe_sum"]
    f1 = ols(tr.total_err_mkt, tr[joint])
    adj = f1.predict(sm.add_constant(te[joint].astype(float), has_constant="add"))
    dm, lo, hi, n = paired_mae_ci(te.total_pts - (te.mkt_total + adj), te.total_err_mkt)
    w, l, p = ou_rate(te.mkt_total + adj, te.mkt_total, te.total_pts)
    print(f"  joint {joint}: train F-test p={f1.f_pvalue:.3f}; coefs " + ", ".join(f"{k}={f1.params[k]:+.3f}(p={f1.pvalues[k]:.2f})" for k in joint))
    print(f"      OOS dMAE vs market {dm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}; O/U {w}-{l}-{p} ({w/max(w+l,1):.3f}); mean|adj|={np.abs(adj).mean():.2f}")

    print("\n-- (C) MARKET-FREE baseline (LEAN) + pace: fit train, OOS 2023-2025 --")
    fb, pb = fit_pred(tr, te, BASE)
    dm, lo, hi, n = paired_mae_ci(te.total_pts - pb, te.total_err_mkt)
    print(f"  BASE alone: test MAE={mae(pb, te.total_pts):.3f} (market {mae(te.mkt_total, te.total_pts):.3f}; dMAE vs mkt {dm:+.3f} [{lo:+.3f},{hi:+.3f}])  bias={np.mean(te.total_pts-pb):+.2f}")
    print(f"  {'+ x':11s} {'coef (se) p [train]':>26s}  {'1 SD of x':>9s}  {'pts per SD':>10s}   test MAE   dMAE vs BASE [95% CI]   O/U vs mkt")
    rows = []
    for c in PACE:
        f, p = fit_pred(tr, te, BASE + [c])
        dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
        w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
        sd = tr[c].std()
        rows.append((c, f.params[c], f.bse[c], f.pvalues[c], sd, f.params[c] * sd, mae(p, te.total_pts), dm, lo, hi))
        print(f"  {c:11s} {f.params[c]:+8.3f} ({f.bse[c]:.3f}) p={f.pvalues[c]:.3f}   {sd:8.3f}   {f.params[c]*sd:+8.2f}   {mae(p, te.total_pts):.3f}   {dm:+.3f} [{lo:+.3f},{hi:+.3f}]   {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
    for lab, cols in [("BASE + spp_avg + ppd_sum", BASE + ["spp_avg", "ppd_sum"]),
                      ("BASE + spp_avg + pr_sum + proe_sum", BASE + ["spp_avg", "pr_sum", "proe_sum"]),
                      ("BASE + all pace/style", BASE + PACE),
                      ("BASE + sppr_avg + plays_sum", BASE + ["sppr_avg", "plays_sum"])]:
        f, p = fit_pred(tr, te, cols)
        dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
        dmm, lom, him, _ = paired_mae_ci(te.total_pts - p, te.total_err_mkt)
        w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
        print(f"  {lab:38s} test MAE={mae(p, te.total_pts):.3f}  dMAE vs BASE {dm:+.3f} [{lo:+.3f},{hi:+.3f}]  vs mkt {dmm:+.3f} [{lom:+.3f},{him:+.3f}]  O/U {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
        print("      coefs: " + ", ".join(f"{k}={f.params[k]:+.3f}(p={f.pvalues[k]:.2f})" for k in cols if k not in BASE))

    print("\n-- (C2) rolling-origin inside the pbp era (fit REG seasons < Y, test Y; Y=2013..2019 PRE-2020, then 2023-25) --")
    print(f"  {'Y':>5s} {'n':>4s} {'BASE':>7s} {'+spp':>7s} {'+spp+ppd':>9s} {'+all':>7s} {'mkt':>7s}   spp coef")
    tot = {"BASE": [], "+spp": [], "+spp+ppd": [], "+all": [], "mkt": []}
    for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
        trY = d[d.season < Y]; teY = d[d.season == Y]
        if Y >= 2023: trY = d[d.season <= 2019] if Y == 2023 else d[(d.season <= 2019) | (d.season < Y)]
        out = {}
        for lab, cols in [("BASE", BASE), ("+spp", BASE + ["spp_avg"]), ("+spp+ppd", BASE + ["spp_avg", "ppd_sum"]), ("+all", BASE + PACE)]:
            f, p = fit_pred(trY, teY, cols); out[lab] = mae(p, teY.total_pts)
            if lab == "+spp": sc = f.params["spp_avg"]
        out["mkt"] = mae(teY.mkt_total, teY.total_pts)
        for k in tot: tot[k].append(out[k])
        print(f"  {Y:5d} {len(teY):4d} {out['BASE']:7.3f} {out['+spp']:7.3f} {out['+spp+ppd']:9.3f} {out['+all']:7.3f} {out['mkt']:7.3f}   {sc:+.3f}")
    print("  mean  " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in tot.items()))

    print("\n-- (D) matchup buckets by neutral seconds/play (league-relative, prior 8 games): FAST = fastest quartile, SLOW = slowest quartile (cutoffs from train) --")
    hq = d.loc[d.train, f"h_spp_neut_{v}"]; lo_q, hi_q = np.percentile(pd.concat([hq, d.loc[d.train, f"a_spp_neut_{v}"]]), [25, 75])
    print(f"  cutoffs: fast <= {lo_q:+.2f} s/play vs league, slow >= {hi_q:+.2f}")
    def cls(x): return np.where(x <= lo_q, "F", np.where(x >= hi_q, "S", "M"))
    d["hc"] = cls(d[f"h_spp_neut_{v}"]); d["ac"] = cls(d[f"a_spp_neut_{v}"])
    d["bucket"] = d.hc + d.ac
    d["bucket"] = d.bucket.replace({"FS": "FS/SF", "SF": "FS/SF", "FM": "F+M", "MF": "F+M", "SM": "S+M", "MS": "S+M"})
    fb_all, _ = fit_pred(d[d.train], d, BASE); d["base_res"] = d.total_pts - _
    for lab, sub in [("TRAIN 2009-19 (BASE residual in-sample)", d[d.train]), ("TEST 2023-25", d[d.test])]:
        print(f"  {lab}")
        gb = sub.groupby("bucket").agg(n=("gid", "size"), total=("total_pts", "mean"), mkt=("mkt_total", "mean"), res_mkt=("total_err_mkt", "mean"),
                                       sd_mkt=("total_err_mkt", "std"), res_base=("base_res", "mean"), sd_base=("base_res", "std"), plays=("h_plays_act", lambda s: np.nan))
        gb["plays"] = sub.groupby("bucket").apply(lambda x: (x.h_plays_act + x.a_plays_act).mean())
        gb["se_mkt"] = gb.sd_mkt / np.sqrt(gb.n); gb["se_base"] = gb.sd_base / np.sqrt(gb.n)
        print(gb[["n", "plays", "total", "mkt", "res_mkt", "se_mkt", "res_base", "se_base"]].round(2).loc[[b for b in ["FF", "F+M", "MM", "FS/SF", "S+M", "SS"] if b in gb.index]].to_string())
    # FF vs SS difference in BASE residual, pooled test
    ff = d[d.test & (d.bucket == "FF")].base_res; ss = d[d.test & (d.bucket == "SS")].base_res
    if len(ff) > 5 and len(ss) > 5:
        t, p = stats.ttest_ind(ff, ss, equal_var=False)
        print(f"  TEST FF vs SS (market-free residual): diff={ff.mean()-ss.mean():+.2f} pts, Welch p={p:.3f}, n={len(ff)}/{len(ss)}")
    ff = d[d.train & (d.bucket == "FF")].base_res; ss = d[d.train & (d.bucket == "SS")].base_res
    t, p = stats.ttest_ind(ff, ss, equal_var=False)
    print(f"  TRAIN FF vs SS (market-free residual, in-sample baseline): diff={ff.mean()-ss.mean():+.2f} pts, Welch p={p:.3f}, n={len(ff)}/{len(ss)}")
    ff = d[d.train & (d.bucket == "FF")].total_err_mkt; ss = d[d.train & (d.bucket == "SS")].total_err_mkt
    t, p = stats.ttest_ind(ff, ss, equal_var=False)
    print(f"  TRAIN FF vs SS (market residual): diff={ff.mean()-ss.mean():+.2f} pts, Welch p={p:.3f}")
