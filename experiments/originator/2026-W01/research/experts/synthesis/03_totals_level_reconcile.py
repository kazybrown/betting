"""Synthesis check 03: reconcile the totals level (totals expert: prior-season realized mean + in-season
blend; market critic: prior-season CLOSE mean / median target; early-season: W1 -0.75) and the
divisional term (totals V3 -1.36*div vs HFA critic mean-preserving -0.75/+0.40) on ONE base:
total = L + 0.30*elo_sum (ratings-only), rolling-origin 2010-2025, REG games with a closing total."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research"); sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.api as sm
from common import build, paired_mae_ci, mae
m = build(K_team=3, K_lg=128, verbose=True)
r = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna() & (m.season >= 2010)].copy()
reg_all = m[(m.game_type == "REG") & m.mkt_total.notna()]
per = reg_all.groupby("season").agg(real_mean=("total_pts", "mean"), real_median=("total_pts", "median"), close_mean=("mkt_total", "mean"))
per["mean_minus_median"] = per.real_mean - per.real_median; per["mean_minus_close"] = per.real_mean - per.close_mean
print("[A] per-season REG totals (mean/median/close):"); print(per.round(2).to_string())
print(f"[A] 2025: realized mean {per.loc[2025,'real_mean']:.2f}, median {per.loc[2025,'real_median']:.1f}, mean close {per.loc[2025,'close_mean']:.2f}")
print(f"[A] 2009-25 avg (mean - median) {per.loc[2009:].mean_minus_median.mean():+.2f}; avg (mean - close) {per.loc[2009:].mean_minus_close.mean():+.2f}; "
      f"2022-25 {per.loc[2022:].mean_minus_median.mean():+.2f} / {per.loc[2022:].mean_minus_close.mean():+.2f}")
prev = per.shift(1)  # prior-season values keyed by season
r["L_mean"] = r.season.map(prev.real_mean); r["L_median"] = r.season.map(prev.real_median); r["L_close"] = r.season.map(prev.close_mean)
r["L_blend"] = r.lg_blend                      # K=128 blend of prior realized mean with season-to-date mean
r["L_blend_m05"] = r.lg_blend - 0.5
r["L_blend_close"] = r.lg_blend - (r.L_mean - r.L_close)
cands = ["L_mean", "L_median", "L_close", "L_mean_m05", "L_blend", "L_blend_m05", "L_blend_close"]
r["L_mean_m05"] = r.L_mean - 0.5
def report(win, lab):
    w = r[(r.season >= win[0]) & (r.season <= win[1])]
    print(f"[B] {lab} n={len(w)} | market MAE {mae(w.mkt_total, w.total_pts):.3f} bias {(w.total_pts-w.mkt_total).mean():+.2f}")
    ref = w.total_pts - (w.L_mean + 0.30 * w.elo_sum)
    for c in cands:
        e = w.total_pts - (w[c] + 0.30 * w.elo_sum)
        d, lo, hi, _ = paired_mae_ci(e, ref)
        print(f"    {c:14s} MAE {np.abs(e).mean():.3f} bias {e.mean():+.2f}  dMAE vs L_mean {d:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"O/U vs close {np.mean((w[c]+0.30*w.elo_sum > w.mkt_total) == (w.total_pts > w.mkt_total)):.3f}")
report((2010, 2021), "ROLLING 2010-21 (prior-season priors, no fitting)")
report((2022, 2025), "ROLLING 2022-25")
# divisional term on the blended ratings-only base
r["base"] = r.L_blend + 0.30 * r.elo_sum
for win, lab in [((2010, 2021), "2010-21"), ((2022, 2025), "2022-25")]:
    w = r[(r.season >= win[0]) & (r.season <= win[1])]
    e = w.total_pts - w.base
    em = w.total_pts - w.mkt_total
    print(f"[C] {lab}: base bias div {e[w['div']==1].mean():+.2f} (n={int((w['div']==1).sum())}) non-div {e[w['div']==0].mean():+.2f}; "
          f"market residual div {em[w['div']==1].mean():+.2f} non-div {em[w['div']==0].mean():+.2f}; div share {w['div'].mean():.2f}")
    for name, dv, nd in [("V3 -1.36/0", -1.36, 0.0), ("mean-pres -0.85/+0.45", -0.85, 0.45), ("HFA-critic -0.75/+0.40", -0.75, 0.40)]:
        adj = np.where(w["div"] == 1, dv, nd)
        e2 = w.total_pts - (w.base + adj)
        d, lo, hi, _ = paired_mae_ci(e2, e)
        print(f"      {name:24s} bias div {e2[w['div']==1].mean():+.2f} non-div {e2[w['div']==0].mean():+.2f} overall {e2.mean():+.2f} | dMAE {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
# week-1 level
for win, lab in [((2010, 2021), "2010-21"), ((2022, 2025), "2022-25"), ((2010, 2025), "2010-25")]:
    w = r[(r.season >= win[0]) & (r.season <= win[1])]
    w1 = w[w.week == 1]; w2 = w[w.week >= 2]
    e1 = w1.total_pts - w1.base
    e1s = w1.total_pts - (w1.base - 0.75)
    d, lo, hi, _ = paired_mae_ci(e1s, e1)
    print(f"[D] {lab} week 1 n={len(w1)}: base bias {e1.mean():+.2f} (se {e1.std()/np.sqrt(len(w1)):.2f}) vs weeks 2+ {(w2.total_pts-w2.base).mean():+.2f}; "
          f"market W1 residual {(w1.total_pts-w1.mkt_total).mean():+.2f}; W1 shift -0.75 -> bias {e1s.mean():+.2f}, dMAE {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
# compact V3 refit (train 2009-21) to confirm coefficient signs/sizes with mean-preserving div and ENV bins
tr = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna() & (m.season >= 2009) & (m.season <= 2021)].copy()
te = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna() & (m.season >= 2022)].copy()
def feats(d):
    X = pd.DataFrame({"elo_sum": d.elo_sum, "gt_dev": d.h_gt + d.a_gt - 2 * d.lg_blend, "qb_sum": d.qb_sum,
                      "div": d["div"], "dome": d.is_dome.astype(int), "wind_c": np.where(d.outdoor == 1, d.wind_f - 8.4, 0.0),
                      "cold20": ((d.outdoor == 1) & (d.temp_f < 20)).astype(int)})
    return sm.add_constant(X)
fit = sm.OLS(tr.total_pts - tr.lg_blend, feats(tr)).fit(cov_type="HC1")
print("[E] V3-style refit on 2009-21 (y = total - lg_blend), coef (se):", {k: f"{v:+.3f} ({fit.bse[k]:.3f})" for k, v in fit.params.items()})
def env_bins(d):
    w = d.wind_f.values; out = d.outdoor.values == 1
    wb = np.select([w <= 5, w <= 9, w <= 14, w <= 19, w <= 24], [0.5, -0.5, -1.5, -2.5, -3.0], -5.0)
    e = np.where(out, wb, 2.0) + np.where(out & (d.temp_f.values < 20), -1.0, 0.0)
    return e
spec = te.lg_blend + 0.10 * te.elo_sum + 0.30 * (te.h_gt + te.a_gt - 2 * te.lg_blend) + 0.72 * te.qb_sum + np.where(te["div"] == 1, -0.85, 0.45) + env_bins(te)
spec_r = te.lg_blend + 0.30 * te.elo_sum + np.where(te["div"] == 1, -0.85, 0.45) + env_bins(te)
spec_r0 = te.lg_blend + 0.30 * te.elo_sum
fitted = te.lg_blend + fit.predict(feats(te))
for lab, p in [("market close", te.mkt_total), ("ratings-only LG+0.30*elo", spec_r0), ("ratings-only + div + ENV", spec_r), ("V3 spec (rounded, no precip)", spec), ("V3 refit coefficients", fitted)]:
    e = te.total_pts - p; d, lo, hi, _ = paired_mae_ci(e, te.total_pts - te.mkt_total)
    print(f"[E] OOS 2022-25 n={len(te)} {lab:30s} MAE {np.abs(e).mean():.3f} RMSE {np.sqrt((e**2).mean()):.2f} bias {e.mean():+.2f} dMAE vs market {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
