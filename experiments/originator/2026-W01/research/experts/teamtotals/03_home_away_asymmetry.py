"""THEORY 2: home/away asymmetry at fixed spread. Does the home team score more of the total
than home = T/2 - S/2 implies?

Algebra: r_home - r_away = margin + S (spread error), r_home + r_away = total error. So an
allocation asymmetry is exactly half the market spread error conditional on S. Tests:
 (a) E[r_home] vs E[r_away] overall and by |S|, train and test
 (b) favorite/dog residuals: does a HOME favorite score more (vs identity) than an AWAY
     favorite of the same size?  r_fav ~ home_fav + |S| + T ; r_dog likewise
 (c) home share of points: home_score/total vs home_tt/T
 (d) neutral-site games
 (e) OOS: does a home-shift (fit on train) lower MAE of home/away team totals in 2022-25?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from common import load, mean_ci, over_rate, boot_ci

g = load(min_season=1999)
g["sp_err"] = g.margin + g.S
g["tot_err"] = g.total_pts - g["T"]
g["home_share"] = g.home_score / g.total_pts
g["home_share_tt"] = g.home_tt / g["T"]
g["share_resid"] = g.home_share - g.home_share_tt

for nm, d in [("TRAIN 1999-2021", g[g.train]), ("TEST 2022-2025", g[g.test]), ("2009-2025", g[g.season >= 2009])]:
    d = d[~d.neutral]
    m1 = mean_ci(d.r_home); m2 = mean_ci(d.r_away); m3 = mean_ci(d.sp_err); m4 = mean_ci(d.share_resid)
    print(f"\n{nm} (non-neutral, n={len(d)}): mean r_home {m1[0]:+.3f} [{m1[1]:+.3f},{m1[2]:+.3f}] | mean r_away {m2[0]:+.3f} [{m2[1]:+.3f},{m2[2]:+.3f}] "
          f"| r_home - r_away = spread err {m3[0]:+.3f} [{m3[1]:+.3f},{m3[2]:+.3f}] | home share resid {m4[0]:+.4f} [{m4[1]:+.4f},{m4[2]:+.4f}]")

# (b) home favorite vs away favorite at the same |S|
tr = g[g.train & ~g.neutral].copy()
te = g[g.test & ~g.neutral].copy()
tr["home_fav_i"] = tr.home_fav.astype(int); te["home_fav_i"] = te.home_fav.astype(int)
for y in ["r_fav", "r_dog", "sp_err", "tot_err"]:
    m = smf.ols(f"{y} ~ home_fav_i + abs_S + T", tr).fit(cov_type="HC1")
    print(f"TRAIN {y:>7s} ~ home_fav + |S| + T : home_fav coef {m.params['home_fav_i']:+.3f} (se {m.bse['home_fav_i']:.3f}, p={m.pvalues['home_fav_i']:.3f}) | |S| {m.params['abs_S']:+.3f} (p={m.pvalues['abs_S']:.3f})")
for y in ["r_fav", "r_dog", "sp_err", "tot_err"]:
    m = smf.ols(f"{y} ~ home_fav_i + abs_S + T", te).fit(cov_type="HC1")
    print(f"TEST  {y:>7s} ~ home_fav + |S| + T : home_fav coef {m.params['home_fav_i']:+.3f} (se {m.bse['home_fav_i']:.3f}, p={m.pvalues['home_fav_i']:.3f}) | |S| {m.params['abs_S']:+.3f} (p={m.pvalues['abs_S']:.3f})")

# by |S| bin: home-fav games vs away-fav games, favorite residual and dog residual
fbins = [-0.01, 0.5, 3, 6.5, 9.5, 13.5, 30]; flab = ["pick", "1-3", "3.5-6.5", "7-9.5", "10-13.5", "14+"]
g["fbin"] = pd.cut(g.abs_S, bins=fbins, labels=flab)
for nm, d in [("TRAIN", g[g.train & ~g.neutral]), ("TEST", g[g.test & ~g.neutral])]:
    print(f"\n{nm}: favorite residual (fav_score - fav_tt) when favorite is HOME vs AWAY, by |S|")
    print(f"{'|S|':>8s} {'n_homefav':>9s} {'r_fav(H)':>9s} {'r_dog(H)':>9s} {'n_awayfav':>9s} {'r_fav(A)':>9s} {'r_dog(A)':>9s} {'diff r_fav':>11s} {'ci':>16s}")
    for b, d2 in d.groupby("fbin", observed=True):
        h = d2[d2.home_fav]; a = d2[~d2.home_fav]
        if len(a) < 5 or len(h) < 5:
            continue
        diff = h.r_fav.mean() - a.r_fav.mean()
        se = np.sqrt(h.r_fav.var(ddof=1) / len(h) + a.r_fav.var(ddof=1) / len(a))
        print(f"{b:>8s} {len(h):9d} {h.r_fav.mean():+9.2f} {h.r_dog.mean():+9.2f} {len(a):9d} {a.r_fav.mean():+9.2f} {a.r_dog.mean():+9.2f} {diff:+11.2f} [{diff-1.96*se:+6.2f},{diff+1.96*se:+6.2f}]")

# (d) neutral site
nn = g[g.neutral]
m1 = mean_ci(nn.r_home); m2 = mean_ci(nn.r_away)
print(f"\nNeutral-site games n={len(nn)}: r_home {m1[0]:+.2f} [{m1[1]:+.2f},{m1[2]:+.2f}]  r_away {m2[0]:+.2f} [{m2[1]:+.2f},{m2[2]:+.2f}]")

# (e) OOS: a home-side shift (train mean of (r_home - r_away)/2), spread-preserving vs not
k = float(((tr.r_home - tr.r_away) / 2).mean())
print(f"\nTRAIN implied home shift k = mean(r_home - r_away)/2 = {k:+.3f}")
for side, y, tt, sgn in [("home", "home_score", "home_tt", +1), ("away", "away_score", "away_tt", -1)]:
    e0 = (te[y] - te[tt]).abs().values
    e1 = (te[y] - (te[tt] + sgn * k)).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"  OOS {side}: MAE identity {e0.mean():.3f} vs identity{sgn*k:+.2f}: {e1.mean():.3f}  dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}]")

# (f) is the asymmetry era-dependent? spread error by season block (drives any home shift)
g["block"] = pd.cut(g.season, [1998, 2004, 2010, 2016, 2021, 2025], labels=["1999-04", "2005-10", "2011-16", "2017-21", "2022-25"])
print("\nSpread error (margin+S) = 2 x home allocation residual, by block (non-neutral):")
d = g[~g.neutral]
print(d.groupby("block", observed=True).sp_err.agg(["size", "mean", "sem"]).round(3).to_string())
