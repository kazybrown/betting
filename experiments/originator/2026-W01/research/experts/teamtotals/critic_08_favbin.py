"""CRITIC of TT5 (favorites 7-9.5 score +1 vs identity). Attacks:
 (a) selection: the '24/27 seasons positive' is for a bin chosen after looking at the era table.
     Compare with the sign count of r_fav OVERALL (the market total under-bias makes r_fav positive
     in most seasons regardless of bin) and run a within-season permutation of bin labels to get
     P(max over 5 bins of positive-season count >= 24) under no bin-specific effect.
 (b) bin-boundary sensitivity of r_fav, tot_err, fav_sp_err in train and test.
 (c) honest rolling 'pick the best bin on prior data, score it on the next season' estimate.
 (d) test-period decomposition (is the spread half there at all in 2022-25?) and whether the
     bin's total error exceeds the period-wide total error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from common import load, mean_ci, boot_ci

g = load(min_season=1999, verbose=False)
g["fav_sp_err"] = g.r_fav - g.r_dog; g["tot_err"] = g.total_pts - g["T"]
g["fbin"] = pd.cut(g.abs_S, [-0.01, 0.5, 3, 6.5, 9.5, 13.5, 30], labels=["pick", "1-3", "3.5-6.5", "7-9.5", "10-13.5", "14+"])
gg = g[g.fbin != "pick"].copy(); gg["fbin"] = gg.fbin.cat.remove_unused_categories()
print("(a) seasons with mean r_fav > 0, by bin and overall, 1999-2025:")
per = gg.groupby(["season", "fbin"], observed=True).r_fav.mean().unstack()
cnt = (per > 0).sum()
allpos = (g.groupby("season").r_fav.mean() > 0).sum()
print("   ", cnt.to_dict(), "| ALL games:", int(allpos), "/ 27 |  r_fav centered on season mean:", ((per.sub(g.groupby('season').r_fav.mean(), axis=0)) > 0).sum().to_dict())
# permutation: shuffle bin labels within season
rng = np.random.default_rng(0); mx = []
lab = gg.fbin.values.copy(); seasons = gg.season.values; rf = gg.r_fav.values
idx_by_season = {s: np.where(seasons == s)[0] for s in np.unique(seasons)}
for _ in range(2000):
    l2 = lab.copy()
    for s, ix in idx_by_season.items():
        l2[ix] = lab[ix][rng.permutation(len(ix))]
    dfp = pd.DataFrame({"season": seasons, "fbin": l2, "r": rf})
    p2 = dfp.groupby(["season", "fbin"]).r.mean().unstack()
    mx.append((p2 > 0).sum().max())
mx = np.array(mx)
print(f"    permutation (bin labels shuffled within season, 2000 reps): P(max over 5 bins of #positive seasons >= 24) = {np.mean(mx >= 24):.3f}; P(>= 23) = {np.mean(mx >= 23):.3f}; median max = {np.median(mx):.0f}")
print("\n(b) bin-boundary sensitivity: r_fav / tot_err / fav_sp_err (mean, n)")
print(f"{'bin':>12s} | {'TRAIN r_fav':>22s} {'tot':>6s} {'fav_sp':>7s} | {'TEST r_fav':>22s} {'tot':>6s} {'fav_sp':>7s}")
for lo_, hi_ in [(7, 9.5), (6.5, 10), (7, 10.5), (7.5, 9.5), (6, 9), (7, 7), (7.5, 9), (8, 10), (3.5, 9.5), (6.5, 13.5)]:
    out = []
    for d in [g[g.train], g[g.test]]:
        b = d[(d.abs_S >= lo_) & (d.abs_S <= hi_)]; m, l, h, _ = mean_ci(b.r_fav)
        out.append(f"{m:+.2f} [{l:+.2f},{h:+.2f}] n={len(b):4d} {b.tot_err.mean():+6.2f} {b.fav_sp_err.mean():+7.2f}")
    print(f"{f'[{lo_},{hi_}]':>12s} | " + " | ".join(out))
print("\n(c) honest rolling selection: for season s (2005-2025) pick the bin with the largest mean r_fav over seasons < s, score that bin in s")
picks = []
for s in range(2005, 2026):
    prior = gg[gg.season < s].groupby("fbin", observed=True).r_fav.mean(); best = prior.idxmax()
    cur = gg[(gg.season == s) & (gg.fbin == best)]
    picks.append({"season": s, "picked": best, "prior_mean": prior.max(), "oos_r_fav": cur.r_fav.mean(), "n": len(cur), "oos_tot": cur.tot_err.mean(), "oos_sp": cur.fav_sp_err.mean()})
Pk = pd.DataFrame(picks)
print(Pk.round(2).to_string(index=False))
w = Pk.n.values; print(f"   pooled OOS r_fav of the picked bin = {np.average(Pk.oos_r_fav, weights=w):+.3f} (n={w.sum()}), positive in {(Pk.oos_r_fav>0).sum()}/{len(Pk)} seasons; 7-9.5 picked in {(Pk.picked=='7-9.5').sum()} of {len(Pk)}")
# season-block bootstrap CI for the pooled OOS r_fav of picked bins
rows = []
for s in range(2005, 2026):
    prior = gg[gg.season < s].groupby("fbin", observed=True).r_fav.mean(); best = prior.idxmax()
    rows.append(gg[(gg.season == s) & (gg.fbin == best)][["season", "r_fav"]])
R = pd.concat(rows); u = R.season.unique(); grp = {s: R.loc[R.season == s, "r_fav"].values for s in u}; bs = []
for _ in range(2000):
    pick = rng.choice(u, len(u), replace=True); bs.append(np.concatenate([grp[s] for s in pick]).mean())
print(f"   season-block bootstrap CI for pooled OOS r_fav: [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
print("\n(d) TEST 2022-25, |S| in [7,9.5]:")
b = g[g.test & (g.abs_S >= 7) & (g.abs_S <= 9.5)]; o = g[g.test & ~((g.abs_S >= 7) & (g.abs_S <= 9.5))]
m, l, h, _ = mean_ci(b.r_fav); mt, lt, ht, _ = mean_ci(b.tot_err); ms, ls, hs, _ = mean_ci(b.fav_sp_err)
print(f"   n={len(b)}: r_fav {m:+.2f} [{l:+.2f},{h:+.2f}] = 0.5*tot_err ({0.5*mt:+.2f}; tot_err {mt:+.2f} [{lt:+.2f},{ht:+.2f}]) + 0.5*fav_sp_err ({0.5*ms:+.2f}; fav_sp_err {ms:+.2f} [{ls:+.2f},{hs:+.2f}])")
d = b.tot_err.mean() - o.tot_err.mean(); se = np.sqrt(b.tot_err.var(ddof=1)/len(b) + o.tot_err.var(ddof=1)/len(o))
print(f"   bin tot_err {b.tot_err.mean():+.2f} vs rest-of-period tot_err {o.tot_err.mean():+.2f} (n={len(o)}): diff {d:+.2f} [{d-1.96*se:+.2f},{d+1.96*se:+.2f}]")
print(f"   fav ATS in bin (fav_sp_err>0 vs <0, pushes out): {(b.fav_sp_err>0).sum()}-{(b.fav_sp_err<0).sum()}-{(b.fav_sp_err==0).sum()}  | TRAIN: ", end="")
bt = g[g.train & (g.abs_S >= 7) & (g.abs_S <= 9.5)]
print(f"{(bt.fav_sp_err>0).sum()}-{(bt.fav_sp_err<0).sum()}-{(bt.fav_sp_err==0).sum()} ({(bt.fav_sp_err>0).sum()/((bt.fav_sp_err!=0).sum()):.3f})")
