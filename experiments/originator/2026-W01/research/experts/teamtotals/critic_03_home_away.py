"""CRITIC of TT2 (home/away asymmetry). The expert's algebra (asymmetry = half the conditional
spread error) is right; attacks: (a) does the spread error scale with T (HFA in points bigger in
high-scoring games -> home share of T is not constant)? (b) home_fav x |S| interaction,
(c) per-season spread error 2022-2025 (is the +0.66 a trend or one season?), (d) home share by
T tercile, train/test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from common import load, mean_ci

g = load(min_season=1999, verbose=False)
g = g[~g.neutral].copy()
g["sp_err"] = g.margin + g.S; g["tot_err"] = g.total_pts - g["T"]; g["home_fav_i"] = g.home_fav.astype(int)
tr, te, mod = g[g.train], g[g.test], g[g.season >= 2009]
print("(a) sp_err ~ T (+ S): does the home team's edge in points grow with the total?")
for nm, d in [("TRAIN 99-21", tr), ("2009-21", mod[mod.train]), ("TEST 22-25", te)]:
    r = smf.ols("sp_err ~ T + S", d).fit(cov_type="HC1")
    print(f"  {nm:>11s} n={len(d)}: T {r.params['T']:+.4f} (se {r.bse['T']:.4f}, p={r.pvalues['T']:.3f}) -> per 10 pts of T {10*r.params['T']:+.2f} pts of margin | S {r.params['S']:+.3f} (t vs 0: {r.params['S']/r.bse['S']:+.2f})")
    r2 = smf.ols("r_home ~ T + S", d).fit(cov_type="HC1"); r3 = smf.ols("r_away ~ T + S", d).fit(cov_type="HC1")
    print(f"              r_home: T {r2.params['T']:+.4f} (p={r2.pvalues['T']:.2f}) | r_away: T {r3.params['T']:+.4f} (p={r3.pvalues['T']:.2f})  [both should be 0 under identity+unbiased market]")
print("\n(b) home_fav x |S| interaction on sp_err and r_fav (train / test)")
for nm, d in [("TRAIN", tr), ("TEST", te)]:
    for y in ["sp_err", "r_fav"]:
        r = smf.ols(f"{y} ~ home_fav_i * abs_S + T", d).fit(cov_type="HC1")
        print(f"  {nm} {y:>6s}: home_fav {r.params['home_fav_i']:+.3f} (p={r.pvalues['home_fav_i']:.2f}), |S| {r.params['abs_S']:+.3f} (p={r.pvalues['abs_S']:.2f}), home_fav:|S| {r.params['home_fav_i:abs_S']:+.3f} (p={r.pvalues['home_fav_i:abs_S']:.2f})")
print("\n(c) spread error (margin+S) by season, 2017-2025 (non-neutral):")
print(g[g.season >= 2017].groupby("season").sp_err.agg(["size", "mean", "sem"]).round(2).T.to_string())
print("\n(d) home share residual (home_score/total - home_tt/T) by T tercile:")
for nm, d in [("TRAIN", tr), ("TEST", te)]:
    d = d.assign(share_res=d.home_score / d.total_pts - d.home_tt / d["T"], Tq=pd.qcut(d["T"], 3, labels=["lowT", "midT", "highT"]))
    d = d[d.total_pts > 0]
    for q, d2 in d.groupby("Tq", observed=True):
        m, lo, hi, _ = mean_ci(d2.share_res); ms, ls, hs, _ = mean_ci(d2.sp_err)
        print(f"  {nm} {q:>5s} n={len(d2)}: share resid {m:+.4f} [{lo:+.4f},{hi:+.4f}] | sp_err {ms:+.2f} [{ls:+.2f},{hs:+.2f}]")
