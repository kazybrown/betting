"""THEORY 1b: the identity gives the MEAN of a team's score; a team total is bet as over/under,
so the fair number is the MEDIAN. Team scores are right-skewed, more so at low expected
scores, so median < mean for low expected totals. Test: P(score > identity) by expected
score bin, quantile (median) regression vs OLS, and an OOS skew-adjustment table.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from common import load, mean_ci, over_rate, boot_ci

g = load(min_season=1999)

# long format: one row per team-game
rows = []
for side, sc, tt in [("home", "home_score", "home_tt"), ("away", "away_score", "away_tt")]:
    d = g[["gid", "season", "train", "test", "S", "T", sc, tt, "home_fav", "abs_S"]].copy()
    d = d.rename(columns={sc: "score", tt: "tt"})
    d["side"] = side
    d["is_fav"] = (d.home_fav if side == "home" else ~d.home_fav)
    rows.append(d)
L = pd.concat(rows, ignore_index=True)
L["resid"] = L.score - L.tt
L["over"] = L.score > L.tt
L["push"] = L.score == L.tt
print(f"team-games: {len(L)}  (train {int(L.train.sum())}, test {int(L.test.sum())})")

# 1. skewness of score residuals and mean-median gap by expected-score bin
ebins = [0, 15, 17.5, 20, 22.5, 25, 27.5, 30, 60]
L["ebin"] = pd.cut(L.tt, ebins)
for nm, d in [("TRAIN 1999-2021", L[L.train]), ("TEST 2022-2025", L[L.test])]:
    print(f"\n{nm}: score vs identity by expected-score (identity) bin")
    print(f"{'tt bin':>14s} {'n':>5s} {'mean_r':>7s} {'ci':>16s} {'median_r':>8s} {'skew':>6s} {'P(over)':>8s} {'ci':>16s} {'push':>5s}")
    for b, d2 in d.groupby("ebin", observed=True):
        m, lo, hi, _ = mean_ci(d2.resid)
        o, nn = over_rate(d2.score, d2.tt)
        se = np.sqrt(o * (1 - o) / nn)
        print(f"{str(b):>14s} {len(d2):5d} {m:+7.2f} [{lo:+6.2f},{hi:+6.2f}] {d2.resid.median():+8.2f} {stats.skew(d2.resid):6.2f} {o:8.3f} [{o-1.96*se:6.3f},{o+1.96*se:6.3f}] {d2.push.mean():5.3f}")

# 2. quantile (median) regression vs OLS on train: score ~ tt   (tt = identity value)
tr, te = L[L.train], L[L.test]
q = smf.quantreg("score ~ tt", tr).fit(q=0.5)
o = smf.ols("score ~ tt", tr).fit()
print("\nTRAIN median regression  score ~ tt :", {k: round(v, 3) for k, v in q.params.items()})
print("TRAIN OLS               score ~ tt :", {k: round(v, 3) for k, v in o.params.items()})
# what the median regression implies at several expected scores
for x in [14, 17, 20, 23, 26, 29, 32]:
    print(f"   identity {x:2d} -> median-reg fair {q.params['Intercept'] + q.params['tt']*x:5.2f}  (OLS {o.params['Intercept'] + o.params['tt']*x:5.2f})")

# 3. simple skew-adjustment table fit on train: shift = median residual in expected-score bin
tab = tr.groupby("ebin", observed=True).resid.median()
print("\nTRAIN median residual by bin (candidate shift table):")
print(tab.round(2).to_string())

def apply_table(d):
    return d.tt + d.ebin.map(tab).astype(float).fillna(0.0)

# candidate rules
def rule_linear(d):
    # continuous version: shift = clip(-0.5 - 0.10*(tt - 21), lower=-1.5, upper=+0.5)?? fit from median reg instead:
    return q.params["Intercept"] + q.params["tt"] * d.tt

def rule_simple(d):
    # piecewise: -1.0 if tt < 17.5; -0.5 if 17.5<=tt<22.5; 0 if 22.5<=tt<27.5; +0.5 if >= 27.5?  (tested below)
    return d.tt + np.select([d.tt < 17.5, d.tt < 22.5, d.tt < 27.5], [-1.0, -0.5, 0.0], 0.0)

print("\nOOS 2022-2025 (team-games n=%d): MAE / P(over) of candidate rules" % len(te))
print(f"{'rule':>34s} {'MAE':>7s} {'dMAE vs ident':>22s} {'P(over)':>8s} {'P(over) tt<20':>14s} {'P(over) tt>=27.5':>16s}")
cands = [("identity", te.tt), ("median-reg linear", rule_linear(te)), ("bin median table", apply_table(te)), ("piecewise -1/-.5/0/0", rule_simple(te)),
         ("identity - 0.5 flat", te.tt - 0.5)]
e0 = (te.score - te.tt).abs().values
for nm, p in cands:
    e = (te.score - p).abs().values
    lo, hi = boot_ci(e0 - e)
    o_all, _ = over_rate(te.score, p)
    lowm = te.tt < 20
    him = te.tt >= 27.5
    o_low, _ = over_rate(te.score[lowm], p[lowm])
    o_hi, _ = over_rate(te.score[him], p[him])
    print(f"{nm:>34s} {e.mean():7.3f} {np.mean(e0-e):+7.3f} [{lo:+6.3f},{hi:+6.3f}] {o_all:8.3f} {o_low:14.3f} {o_hi:16.3f}")

# 4. rolling-origin check of P(over) for low expected totals: every season from 2009
print("\nP(over identity) for tt<20 vs tt>=25, by season (robustness):")
for s, d in L[L.season >= 2009].groupby("season"):
    lo_, n1 = over_rate(d.score[d.tt < 20], d.tt[d.tt < 20])
    hi_, n2 = over_rate(d.score[d.tt >= 25], d.tt[d.tt >= 25])
    print(f"  {s}: tt<20 P(over)={lo_:.3f} (n={n1})   tt>=25 P(over)={hi_:.3f} (n={n2})")
lowall = L[(L.season >= 2009) & (L.tt < 20)]
o, n = over_rate(lowall.score, lowall.tt)
print(f"2009-2025 tt<20: P(over)={o:.3f} n={n}  binomial p={stats.binomtest(int(o*n), n, 0.5).pvalue:.4f}")
hiall = L[(L.season >= 2009) & (L.tt >= 25)]
o, n = over_rate(hiall.score, hiall.tt)
print(f"2009-2025 tt>=25: P(over)={o:.3f} n={n}  binomial p={stats.binomtest(int(o*n), n, 0.5).pvalue:.4f}")
