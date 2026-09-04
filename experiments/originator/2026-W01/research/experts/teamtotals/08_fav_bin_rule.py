"""Follow-up to 01/06: the only era-stable pattern in the bins was favorites of 7-9.5 scoring
~+1 above identity in every era block (and the total in those games going over by ~+1).
Test a concrete rule OOS: for 7 <= |S| <= 9.5, favorite team total +0.5 (dog unchanged),
fit magnitude on train (1999-2021), evaluate 2022-2025. Also the monotone alternative
(fav += 0.05*|S|) to show the effect is not monotone in |S|."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from common import load, mean_ci, over_rate, boot_ci

g = load(min_season=1999, verbose=False)
tr, te = g[g.train], g[g.test]
b_tr = tr[(tr.abs_S >= 7) & (tr.abs_S <= 9.5)]; b_te = te[(te.abs_S >= 7) & (te.abs_S <= 9.5)]
m, lo, hi, _ = mean_ci(b_tr.r_fav)
print(f"TRAIN |S| 7-9.5: n={len(b_tr)} r_fav {m:+.2f} [{lo:+.2f},{hi:+.2f}] median {b_tr.r_fav.median():+.2f} P(fav over) {over_rate(b_tr.fav_score, b_tr.fav_tt)[0]:.3f}; r_dog {b_tr.r_dog.mean():+.2f}")
m, lo, hi, _ = mean_ci(b_te.r_fav)
o, n = over_rate(b_te.fav_score, b_te.fav_tt)
print(f"TEST  |S| 7-9.5: n={len(b_te)} r_fav {m:+.2f} [{lo:+.2f},{hi:+.2f}] median {b_te.r_fav.median():+.2f} P(fav over) {o:.3f} (binom p={stats.binomtest(int(round(o*n)), n, 0.5).pvalue:.3f}); r_dog {b_te.r_dog.mean():+.2f}")
for shift in [0.5, 1.0]:
    e0 = np.abs(b_te.fav_score - b_te.fav_tt).values; e1 = np.abs(b_te.fav_score - (b_te.fav_tt + shift)).values
    lo, hi = boot_ci(e0 - e1)
    print(f"  OOS rule fav_tt += {shift} in 7-9.5: MAE {e0.mean():.3f} -> {e1.mean():.3f}, dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(b_te)}); P(fav over) after shift {over_rate(b_te.fav_score, b_te.fav_tt+shift)[0]:.3f}")
# per-season sign check of r_fav in the bin
print("r_fav in 7-9.5 by season:", g[(g.abs_S >= 7) & (g.abs_S <= 9.5)].groupby("season").r_fav.mean().round(2).to_dict())
pos = (g[(g.abs_S >= 7) & (g.abs_S <= 9.5)].groupby("season").r_fav.mean() > 0)
print(f"seasons with r_fav > 0: {pos.sum()}/{len(pos)} (binomial p={stats.binomtest(int(pos.sum()), len(pos), 0.5).pvalue:.3f})")
# monotone alternative
for k in [0.03, 0.05]:
    e0 = np.abs(te.fav_score - te.fav_tt).values; e1 = np.abs(te.fav_score - (te.fav_tt + k * te.abs_S)).values
    lo, hi = boot_ci(e0 - e1)
    print(f"  OOS monotone rule fav_tt += {k}*|S| (all games): dMAE {np.mean(e0-e1):+.4f} [{lo:+.4f},{hi:+.4f}] n={len(te)}")
# decomposition: how much of r_fav in the bin is total error vs spread error (train, all eras)
b = g[(g.abs_S >= 7) & (g.abs_S <= 9.5)]
print(f"ALL 1999-2025 |S| 7-9.5 n={len(b)}: r_fav {b.r_fav.mean():+.2f} = 0.5*total_err ({0.5*b.total_err_mkt.mean():+.2f}) + 0.5*fav_spread_err ({0.5*(b.r_fav-b.r_dog).mean():+.2f}); r_dog {b.r_dog.mean():+.2f}")
