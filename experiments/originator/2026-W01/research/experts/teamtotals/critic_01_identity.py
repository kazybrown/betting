"""CRITIC of TT1 (identity linearity). Re-derives key numbers, then attacks:
 (a) slope-only Wald tests for fav/dog on 1999-2021 (expert only reported these with the intercept)
     and the favorite-margin slope vs 1.0 (is there a favorite/dog asymmetry hiding as a spread bias?)
 (b) Huber robust regression of the slopes (outlier-robust alternative spec)
 (c) SEASON-BLOCK bootstrap of the rolling-origin pooled dMAE (games within a season share a
     bias; the expert's game-level bootstrap treats them as independent -> too-narrow CIs)
 (d) modern-only expanding window (fit from 2009, score 2014-2025): identity vs OLS / OLS-nonlinear
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from common import load, boot_ci

g = load(min_season=1999)
tr, te = g[g.train], g[g.test]
print(f"train n={len(tr)} test n={len(te)}")

print("\n(a) slope-only Wald tests and per-coefficient t-tests vs identity, TRAIN 1999-2021 (HC1)")
for y, X, hyp, ident in [("home_score", ["S", "T"], "S = -0.5, T = 0.5", {"S": -0.5, "T": 0.5}),
                         ("away_score", ["S", "T"], "S = 0.5, T = 0.5", {"S": 0.5, "T": 0.5}),
                         ("fav_score", ["abs_S", "T"], "abs_S = 0.5, T = 0.5", {"abs_S": 0.5, "T": 0.5}),
                         ("dog_score", ["abs_S", "T"], "abs_S = -0.5, T = 0.5", {"abs_S": -0.5, "T": 0.5})]:
    r = sm.OLS(tr[y], sm.add_constant(tr[X])).fit(cov_type="HC1")
    w = r.wald_test(hyp, scalar=True)
    ts = {k: (r.params[k] - ident[k]) / r.bse[k] for k in X}
    print(f"  {y:>10s}: " + ", ".join(f"{k} {r.params[k]:+.3f} (t vs ident {ts[k]:+.2f}, p={2*stats.norm.sf(abs(ts[k])):.3f})" for k in X) + f" | joint slopes p={float(w.pvalue):.3f}")
# favorite margin slope: fav_score - dog_score ~ abs_S + T; identity says slope 1.0 on |S|, 0 on T
tr2 = tr.assign(fav_margin=tr.fav_score - tr.dog_score)
r = smf.ols("fav_margin ~ abs_S + T", tr2).fit(cov_type="HC1")
print(f"  fav_margin ~ |S| + T: |S| slope {r.params['abs_S']:+.3f} (se {r.bse['abs_S']:.3f}, t vs 1.0 = {(r.params['abs_S']-1)/r.bse['abs_S']:+.2f}), T {r.params['T']:+.3f} (p={r.pvalues['T']:.2f}), const {r.params['Intercept']:+.2f}")
print("   -> any |S| slope != 1 is a SPREAD bias (favorites cover more/less per point), not an allocation error; the split at fixed (S,T) is still T/2 +/- S/2 of whatever S is")
te2 = te.assign(fav_margin=te.fav_score - te.dog_score)
r = smf.ols("fav_margin ~ abs_S + T", te2).fit(cov_type="HC1")
print(f"  TEST fav_margin ~ |S| + T: |S| slope {r.params['abs_S']:+.3f} (se {r.bse['abs_S']:.3f}), T {r.params['T']:+.3f} (p={r.pvalues['T']:.2f}) n={len(te2)}")

print("\n(b) Huber robust regression (RLM) of the slopes, TRAIN 1999-2021")
for y, X in [("home_score", ["S", "T"]), ("away_score", ["S", "T"]), ("fav_score", ["abs_S", "T"]), ("dog_score", ["abs_S", "T"])]:
    r = sm.RLM(tr[y], sm.add_constant(tr[X]), M=sm.robust.norms.HuberT()).fit()
    print(f"  {y:>10s}: " + ", ".join(f"{k} {r.params[k]:+.3f} (se {r.bse[k]:.3f})" for k in r.params.index))

print("\n(c) rolling origin 2010-2025 (fit on all prior seasons from 1999) with SEASON-BLOCK bootstrap")
g["S_abs"] = g.S.abs(); g["S_sq"] = g.S * g.S_abs; g["ST"] = g.S * g["T"]
rows = []
for s in range(2010, 2026):
    a, b = g[g.season < s], g[g.season == s].copy()
    for side, y, tt in [("h", "home_score", "home_tt"), ("a", "away_score", "away_tt")]:
        p_ols = smf.ols(f"{y} ~ S + T", a).fit().predict(b)
        p_nl = smf.ols(f"{y} ~ S + T + S_abs + S_sq + ST", a).fit().predict(b)
        b[f"{side}_d_ols"] = np.abs(b[y] - b[tt]) - np.abs(b[y] - p_ols)   # identity - ols (negative = identity better)
        b[f"{side}_d_nl"] = np.abs(b[y] - b[tt]) - np.abs(b[y] - p_nl)
    rows.append(b)
P = pd.concat(rows)
rng = np.random.default_rng(0)
seasons = P.season.unique()
for col in ["h_d_ols", "h_d_nl", "a_d_ols", "a_d_nl"]:
    d = P[col].values
    per = P.groupby("season")[col].mean()
    # season block bootstrap: resample seasons with replacement, pooled game-weighted mean
    bs = []
    grp = {s: P.loc[P.season == s, col].values for s in seasons}
    for _ in range(2000):
        pick = rng.choice(seasons, len(seasons), replace=True)
        v = np.concatenate([grp[s] for s in pick]); bs.append(v.mean())
    lo_g, hi_g = boot_ci(d)
    print(f"  {col}: pooled dMAE {d.mean():+.4f} | game-bootstrap CI [{lo_g:+.4f},{hi_g:+.4f}] | season-block CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] | identity better in {(per<0).sum()}/16 seasons")

print("\n(d) modern-only expanding window: fit 2009..s-1, score s (2014-2025)")
gm = g[g.season >= 2009]
rows = []
for s in range(2014, 2026):
    a, b = gm[gm.season < s], gm[gm.season == s].copy()
    for side, y, tt in [("h", "home_score", "home_tt"), ("a", "away_score", "away_tt")]:
        p_ols = smf.ols(f"{y} ~ S + T", a).fit().predict(b)
        p_nl = smf.ols(f"{y} ~ S + T + S_abs + S_sq + ST", a).fit().predict(b)
        b[f"{side}_d_ols"] = np.abs(b[y] - b[tt]) - np.abs(b[y] - p_ols)
        b[f"{side}_d_nl"] = np.abs(b[y] - b[tt]) - np.abs(b[y] - p_nl)
    rows.append(b)
P = pd.concat(rows)
for col in ["h_d_ols", "h_d_nl", "a_d_ols", "a_d_nl"]:
    d = P[col].values; per = P.groupby("season")[col].mean()
    lo, hi = boot_ci(d)
    print(f"  {col}: pooled n={len(d)} dMAE (identity - alt) {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] | identity better in {(per<0).sum()}/{len(per)} seasons | paired-t over seasons p={stats.ttest_1samp(per,0).pvalue:.3f}")

# what the market total shrinkage looks like (drives the T slope < 0.5): total_pts ~ T
r = smf.ols("total_pts ~ T", tr).fit(cov_type="HC1")
print(f"\n(e) market total attenuation, TRAIN: E[total|T] = {r.params['Intercept']:+.2f} + {r.params['T']:.3f} T  (slope se {r.bse['T']:.3f}); TEST: ", end="")
r2 = smf.ols("total_pts ~ T", te).fit(cov_type="HC1")
print(f"{r2.params['Intercept']:+.2f} + {r2.params['T']:.3f} T (se {r2.bse['T']:.3f})")
