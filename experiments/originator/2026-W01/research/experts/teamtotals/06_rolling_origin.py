"""Rolling-origin robustness: for each season 2010-2025 fit allocation formulas on ALL prior
seasons (from 1999) and score that season. Compares identity vs OLS-linear vs OLS-nonlinear
vs median (quantile) regression vs identity + train-mean shift, for home and away team totals.
Also reports the favorite-perspective spread error (r_fav - r_dog) by |S| bin across eras."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import load, boot_ci

g = load(min_season=1999, verbose=False)
g["S_abs"] = g.S.abs(); g["S_sq"] = g.S * g.S_abs; g["ST"] = g.S * g["T"]
out = []
for s in range(2010, 2026):
    tr, te = g[g.season < s], g[g.season == s]
    rec = {"season": s, "n": len(te)}
    for side, y, tt in [("h", "home_score", "home_tt"), ("a", "away_score", "away_tt")]:
        preds = {"identity": te[tt].values,
                 "shift": (te[tt] + (tr[y] - tr[tt]).mean()).values,
                 "ols": smf.ols(f"{y} ~ S + T", tr).fit().predict(te).values,
                 "ols_nl": smf.ols(f"{y} ~ S + T + S_abs + S_sq + ST", tr).fit().predict(te).values,
                 "qreg": smf.quantreg(f"{y} ~ S + T", tr).fit(q=0.5).predict(te).values}
        for k, p in preds.items():
            rec[f"{side}_{k}"] = np.abs(te[y].values - p).mean()
    out.append(rec)
R = pd.DataFrame(out).set_index("season")
print("Rolling-origin MAE by season (h_=home team total, a_=away):")
print(R.round(3).to_string())
print("\nMean over 16 seasons and #seasons identity beats each alternative:")
for side in ["h", "a"]:
    for k in ["shift", "ols", "ols_nl", "qreg"]:
        d = R[f"{side}_identity"] - R[f"{side}_{k}"]
        print(f"  {side} identity - {k:>7s}: mean dMAE {d.mean():+.4f} (sd {d.std():.3f}); identity better in {(d<0).sum()}/16 seasons; paired t p={__import__('scipy').stats.ttest_1samp(d,0).pvalue:.3f}")

# game-level pooled comparison 2010-2025 (all OOS by construction)
pool = []
for s in range(2010, 2026):
    tr, te = g[g.season < s], g[g.season == s].copy()
    for side, y, tt in [("h", "home_score", "home_tt"), ("a", "away_score", "away_tt")]:
        te[f"{side}_e_id"] = np.abs(te[y] - te[tt])
        te[f"{side}_e_nl"] = np.abs(te[y] - smf.ols(f"{y} ~ S + T + S_abs + S_sq + ST", tr).fit().predict(te))
        te[f"{side}_e_q"] = np.abs(te[y] - smf.quantreg(f"{y} ~ S + T", tr).fit(q=0.5).predict(te))
    pool.append(te)
P = pd.concat(pool)
for side in ["h", "a"]:
    for k in ["nl", "q"]:
        d = (P[f"{side}_e_id"] - P[f"{side}_e_{k}"]).values
        lo, hi = boot_ci(d)
        print(f"pooled 2010-2025 (n={len(P)}) {side}: MAE identity - {k}: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

# favorite-perspective spread error by |S| bin and era (is 'big favorites score more' stable?)
g["fav_sp_err"] = g.r_fav - g.r_dog     # = favorite margin - |S|  (>0: favorite covered)
g["fbin"] = pd.cut(g.S_abs, [-0.01, 0.5, 3, 6.5, 9.5, 13.5, 30], labels=["pick", "1-3", "3.5-6.5", "7-9.5", "10-13.5", "14+"])
g["block"] = pd.cut(g.season, [1998, 2004, 2010, 2016, 2021, 2025], labels=["1999-04", "2005-10", "2011-16", "2017-21", "2022-25"])
print("\nFavorite-perspective spread error (fav margin - |S|) by |S| bin x era: mean (n)")
tab = g.pivot_table(index="fbin", columns="block", values="fav_sp_err", aggfunc=["mean", "size"], observed=True)
print(tab["mean"].round(2).astype(str).add(" (" + tab["size"].astype(str) + ")").to_string())
print("\nr_fav (favorite score - identity) by |S| bin x era:")
tab = g.pivot_table(index="fbin", columns="block", values="r_fav", aggfunc="mean", observed=True)
print(tab.round(2).to_string())
print("\ntotal error by |S| bin x era:")
tab = g.pivot_table(index="fbin", columns="block", values="total_err_mkt", aggfunc="mean", observed=True)
print(tab.round(2).to_string())
