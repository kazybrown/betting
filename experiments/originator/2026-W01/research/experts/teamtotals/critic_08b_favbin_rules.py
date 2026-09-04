"""CRITIC follow-up on TT5: the mean r_fav in the 7-9.5 bin is OOS-robust under honest rolling
selection (critic_08). But team totals / spreads are priced at the MEDIAN, so test the actionable
rules in rolling-origin form 2010-2025 (bin fixed a priori at [7,9.5] - it is what the honest
selection picks from 2010 on):
 (a) excess of bin r_fav over the season-wide r_fav (is it bin-specific or the general total bias?)
 (b) fav_tt += 0.5 / +1.0 in the bin: pooled OOS MAE change (season-block CI), P(fav over)
 (c) total += 0.5 / +1.0 in the bin: total MAE change; P(total over T) in the bin by era
 (d) favorite ATS record in the bin by era (the 'favorite lean' recommendation), median r_fav / fav_sp_err
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from common import load, mean_ci, over_rate

g = load(min_season=1999, verbose=False)
g["fav_sp_err"] = g.r_fav - g.r_dog; g["tot_err"] = g.total_pts - g["T"]
inb = (g.abs_S >= 7) & (g.abs_S <= 9.5)
rng = np.random.default_rng(0)
def sb_ci(vals, seasons, n=2000):
    u = np.unique(seasons); grp = {s: vals[seasons == s] for s in u}; bs = []
    for _ in range(n):
        pick = rng.choice(u, len(u), replace=True); bs.append(np.concatenate([grp[s] for s in pick]).mean())
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

oos = g[(g.season >= 2010) & inb].copy()
sm_ = g[g.season >= 2010].groupby("season").r_fav.mean()
oos["excess"] = oos.r_fav - oos.season.map(sm_)
print(f"(a) 2010-2025 bin n={len(oos)}: r_fav {oos.r_fav.mean():+.3f} season-block CI {tuple(round(x,3) for x in sb_ci(oos.r_fav.values, oos.season.values))}; "
      f"excess over season-wide r_fav {oos.excess.mean():+.3f} CI {tuple(round(x,3) for x in sb_ci(oos.excess.values, oos.season.values))}; median r_fav {oos.r_fav.median():+.2f}; P(fav over) {over_rate(oos.fav_score, oos.fav_tt)[0]:.3f}")
print("    components: tot_err {:+.2f} (season-wide {:+.2f}), fav_sp_err {:+.2f}; median tot_err {:+.1f}, median fav_sp_err {:+.1f}".format(
    oos.tot_err.mean(), g[g.season >= 2010].tot_err.mean(), oos.fav_sp_err.mean(), oos.tot_err.median(), oos.fav_sp_err.median()))

print("\n(b) rolling OOS 2010-2025, rule fav_tt += k in the bin (dog unchanged): MAE of the favorite team total")
for k in [0.5, 1.0]:
    e0 = np.abs(oos.fav_score - oos.fav_tt).values; e1 = np.abs(oos.fav_score - (oos.fav_tt + k)).values; d = e0 - e1
    per = pd.Series(d, index=oos.season.values).groupby(level=0).mean()
    print(f"   k={k}: dMAE (identity - rule; + = rule better) {d.mean():+.4f} season-block CI {tuple(round(x,4) for x in sb_ci(d, oos.season.values))} | rule better in {(per>0).sum()}/{len(per)} seasons | P(fav over rule) {over_rate(oos.fav_score, oos.fav_tt + k)[0]:.3f} | per-game-overall equivalent {d.mean()*len(oos)/len(g[g.season>=2010]):+.4f}")
print("\n(c) rolling OOS 2010-2025, rule total += k in the bin: MAE of the game total")
for k in [0.5, 1.0, 1.5]:
    e0 = np.abs(oos.total_pts - oos["T"]).values; e1 = np.abs(oos.total_pts - (oos["T"] + k)).values; d = e0 - e1
    per = pd.Series(d, index=oos.season.values).groupby(level=0).mean()
    print(f"   k={k}: dMAE {d.mean():+.4f} season-block CI {tuple(round(x,4) for x in sb_ci(d, oos.season.values))} | better in {(per>0).sum()}/{len(per)} seasons | P(over T+k) {over_rate(oos.total_pts, oos['T'] + k)[0]:.3f}")
g["block"] = pd.cut(g.season, [1998, 2004, 2010, 2016, 2021, 2025], labels=["1999-04", "2005-10", "2011-16", "2017-21", "2022-25"])
print("\n    P(total over T) in bin vs outside bin, by era:")
for b, d in g.groupby("block", observed=True):
    i, o = d[inb.loc[d.index]], d[~inb.loc[d.index]]
    oi, ni = over_rate(i.total_pts, i["T"]); oo, no = over_rate(o.total_pts, o["T"])
    print(f"      {b}: in-bin P(over) {oi:.3f} (n={ni}) | outside {oo:.3f} (n={no}) | in-bin mean tot_err {i.tot_err.mean():+.2f} vs outside {o.tot_err.mean():+.2f}")
print("\n(d) favorite ATS in the bin by era (fav_sp_err > 0 = favorite covered), and medians:")
for b, d in g[inb].groupby("block", observed=True):
    w, l, p = (d.fav_sp_err > 0).sum(), (d.fav_sp_err < 0).sum(), (d.fav_sp_err == 0).sum()
    print(f"      {b}: fav ATS {w}-{l}-{p} ({w/(w+l):.3f}) | mean fav_sp_err {d.fav_sp_err.mean():+.2f} median {d.fav_sp_err.median():+.1f} | mean r_fav {d.r_fav.mean():+.2f} median {d.r_fav.median():+.2f} | P(fav over tt) {over_rate(d.fav_score, d.fav_tt)[0]:.3f}")
d = g[inb & (g.season >= 2010)]
w, l = (d.fav_sp_err > 0).sum(), (d.fav_sp_err < 0).sum()
print(f"      2010-2025 pooled: fav ATS {w}-{l} ({w/(w+l):.3f}, binomial p={stats.binomtest(int(w), int(w+l), 0.5).pvalue:.3f}) -> a 'favorite lean' in the bin would have LOST ATS; the +1 mean is a right-tail (blowout) effect")
# skew check
print(f"      skew of fav_sp_err in bin (2010-25): {stats.skew(d.fav_sp_err):.2f}; mean-median gap {d.fav_sp_err.mean()-d.fav_sp_err.median():+.2f}")
