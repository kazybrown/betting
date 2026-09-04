"""THEORY 1: implied_total = 46.0 + 0.35*(home_pts_vs_avg + away_pts_vs_avg) [nfelo Elo].
(a) Is b ~ 0.35? Fit total_pts ~ a + b*elo_sum on 2009-2021, test 2022-2025.
(b) Does combined strength predict totals at all, and does the MARKET already price it
    (regress market residual on elo_sum)?
(c) Offense / defense split: rolling points-for / points-against proxies vs elo_sum.
All fits: REG season games with a market total. OOS = seasons 2022-2025 (fit <= 2021)
plus a rolling-origin check (fit on all seasons < Y, test Y).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from common import build, mae, paired_mae_ci, ou_rate

m = build(verbose=True)
d = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna()].copy()
tr, te = d[d.train], d[d.test]
print(f"\nsample: REG games with Elo: train 2009-2021 n={len(tr)}, test 2022-2025 n={len(te)}")

# ---------- (b) does elo_sum predict totals? does the market already price it? ----------
print("\n== (b) predictive content of elo_sum (combined strength, points vs avg) ==")
for lab, x in [("train", tr), ("test", te)]:
    r1 = stats.pearsonr(x.elo_sum, x.total_pts); r2 = stats.pearsonr(x.elo_sum, x.mkt_total); r3 = stats.pearsonr(x.elo_sum, x.total_err_mkt)
    print(f"  {lab:5s} corr(elo_sum,total_pts)={r1[0]:+.3f} (p={r1[1]:.1e}) | corr(elo_sum,mkt_total)={r2[0]:+.3f} | "
          f"corr(elo_sum, total - mkt_total)={r3[0]:+.3f} (p={r3[1]:.2f})  <- ~0 means market already prices it")
mm = smf.ols("mkt_total ~ elo_sum", data=tr).fit(cov_type="HC1")
print(f"  market's own slope: mkt_total = {mm.params['Intercept']:.2f} + {mm.params['elo_sum']:.3f}*elo_sum (se {mm.bse['elo_sum']:.3f}) [train]")

# ---------- (a) slope b ----------
print("\n== (a) total_pts ~ a + b*elo_sum ==")
f1 = smf.ols("total_pts ~ elo_sum", data=tr).fit(cov_type="HC1")
ci = f1.conf_int().loc["elo_sum"]
print(f"  FIT 2009-2021: a={f1.params['Intercept']:.2f}  b={f1.params['elo_sum']:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]  R2={f1.rsquared:.4f}  (spec uses b=0.35, a=46.0)")
f1s = smf.ols("total_pts ~ elo_sum + C(season)", data=tr).fit(cov_type="HC1")
ci = f1s.conf_int().loc["elo_sum"]
print(f"  FIT with season FE:               b={f1s.params['elo_sum']:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]")
f1t = smf.ols("total_pts ~ elo_sum + C(season)", data=te).fit(cov_type="HC1")
ci = f1t.conf_int().loc["elo_sum"]
print(f"  TEST 2022-2025 (in-sample there): b={f1t.params['elo_sum']:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]")
# asymmetric home / away
f2 = smf.ols("total_pts ~ home_pts_vs_avg + away_pts_vs_avg + C(season)", data=tr).fit(cov_type="HC1")
print(f"  split home/away: b_home={f2.params['home_pts_vs_avg']:.3f} (se {f2.bse['home_pts_vs_avg']:.3f})  b_away={f2.params['away_pts_vs_avg']:.3f} (se {f2.bse['away_pts_vs_avg']:.3f})")
# curvature: elo_sum^2 and |elo_dif| (mismatch)
f3 = smf.ols("total_pts ~ elo_sum + I(elo_sum**2) + I(abs(elo_dif_pts)) + C(season)", data=tr).fit(cov_type="HC1")
print(f"  curvature: elo_sum={f3.params['elo_sum']:.3f}  elo_sum^2={f3.params['I(elo_sum ** 2)']:.4f} (p={f3.pvalues['I(elo_sum ** 2)']:.2f})  "
      f"|elo_dif|={f3.params['I(abs(elo_dif_pts))']:.3f} (p={f3.pvalues['I(abs(elo_dif_pts))']:.2f})")

# ---------- OOS 2022-2025 ----------
print("\n== OOS 2022-2025 totals MAE (n=%d) ==" % len(te))
b_fit = f1s.params["elo_sum"]
preds = {
    "market close": te.mkt_total,
    "spec: 46.0 + 0.35*elo_sum": 46.0 + 0.35 * te.elo_sum,
    "lg_prev + 0.35*elo_sum": te.lg_prev + 0.35 * te.elo_sum,
    f"lg_prev + {b_fit:.2f}*elo_sum (fit b)": te.lg_prev + b_fit * te.elo_sum,
    f"lg_blend + {b_fit:.2f}*elo_sum": te.lg_blend + b_fit * te.elo_sum,
    "lg_prev only (no team info)": te.lg_prev,
    "lg_blend only": te.lg_blend,
    "constant 46.0": pd.Series(46.0, index=te.index),
}
for k, p in preds.items():
    e = p - te.total_pts
    dm, lo, hi, n = paired_mae_ci(p - te.total_pts, te.mkt_total - te.total_pts)
    w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
    print(f"  {k:36s} MAE={mae(p, te.total_pts):.3f} bias={e.mean():+.2f}  dMAE vs mkt={dm:+.3f} [{lo:+.3f},{hi:+.3f}]  O/U vs mkt {w}-{l}-{pu} ({w/max(w+l,1):.3f})")

# rolling origin: fit on seasons < Y
print("\n== rolling-origin b and OOS MAE by test season (fit on all REG seasons < Y) ==")
for Y in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
    a = d[d.season < Y]; b = d[d.season == Y]
    f = smf.ols("total_pts ~ elo_sum + C(season)", data=a).fit()
    bb = f.params["elo_sum"]
    p = b.lg_prev + bb * b.elo_sum
    print(f"  {Y}: b={bb:.3f}  MAE lg_prev+b*elo={mae(p, b.total_pts):.3f}  spec(46+.35)={mae(46 + .35*b.elo_sum, b.total_pts):.3f}  market={mae(b.mkt_total, b.total_pts):.3f}  n={len(b)}  realized mean={b.total_pts.mean():.1f} lg_prev={b.lg_prev.iloc[0]:.1f}")

# ---------- (c) offense / defense split ----------
print("\n== (c) offense/defense proxies (blended rolling PF/PA, K=6) vs combined Elo ==")
print("  pf_sum = h_pf + a_pf (both offenses), pa_sum = h_pa + a_pa (both defenses, higher = worse D)")
for lab, form in [("elo only", "total_pts ~ elo_sum + C(season)"),
                  ("pf+pa", "total_pts ~ pf_sum + pa_sum + C(season)"),
                  ("gt (team game-total avg)", "total_pts ~ gt_avg + C(season)"),
                  ("pf+pa+elo", "total_pts ~ pf_sum + pa_sum + elo_sum + C(season)"),
                  ("matchup: h_off_vs_a_def + a_off_vs_h_def", "total_pts ~ h_off_vs_a_def + a_off_vs_h_def + C(season)"),
                  ("prior-season only pf+pa", "total_pts ~ I(h_pf_prev + a_pf_prev) + I(h_pa_prev + a_pa_prev) + C(season)")]:
    f = smf.ols(form, data=tr).fit(cov_type="HC1")
    coefs = {k: f"{v:.3f}" for k, v in f.params.items() if not k.startswith("C(") and k != "Intercept"}
    # OOS: predict test with train slopes; replace season FE with lg_prev offset
    # (season FE absorbs the level; use lg_prev as the level in test)
    Xn = [k for k in f.params.index if not k.startswith("C(") and k != "Intercept"]
    ff = smf.ols(form.replace(" + C(season)", "") + " - 1", data=tr.assign(total_pts=tr.total_pts - tr.lg_prev)).fit()
    p = te.lg_prev.values + sum(ff.params[k] * te.eval(k.replace("I(", "(")).values for k in ff.params.index)
    dm, lo, hi, n = paired_mae_ci(p - te.total_pts, te.mkt_total - te.total_pts)
    print(f"  {lab:42s} train-R2={f.rsquared:.4f} coefs={coefs} | OOS MAE={mae(p, te.total_pts):.3f} dMAE vs mkt={dm:+.3f} [{lo:+.3f},{hi:+.3f}]")

# does the market residual load on PF/PA proxies beyond elo?
print("\n== does the MARKET miss offense/defense structure? residual (total - mkt) regressions ==")
for lab, form in [("elo_sum", "total_err_mkt ~ elo_sum"), ("pf_sum + pa_sum", "total_err_mkt ~ pf_sum + pa_sum"), ("gt_avg", "total_err_mkt ~ gt_avg"),
                  ("lg_prev - mkt level", "total_err_mkt ~ I(lg_prev - mkt_total)")]:
    for lab2, x in [("train", tr), ("test", te)]:
        f = smf.ols(form, data=x).fit(cov_type="HC1")
        coefs = {k: f"{v:+.3f} (p={f.pvalues[k]:.2f})" for k, v in f.params.items() if k != "Intercept"}
        print(f"  {lab:20s} {lab2:5s} {coefs}")
