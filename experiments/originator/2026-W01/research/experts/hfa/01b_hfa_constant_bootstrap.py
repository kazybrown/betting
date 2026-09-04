"""THEORY 1 (cont): paired bootstrap CIs for OOS (2022-2025) MAE differences between HFA constants,
median residual, and a rating-free check (market-implied HFA vs realized) so the k choice is not
an artefact of nfelo's rating scale."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import merged
rng = np.random.default_rng(7)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m = m.dropna(subset=["rating_dif","mkt_spread"])
test = m[m.season>=2022].copy(); n = len(test)
print("OOS n =", n)
r = test.margin - test.rating_dif
print("rating-HFA residual 2022-2025: mean=%.2f median=%.2f  (median of margin - rating_dif)" % (r.mean(), r.median()))
print("  by season median:", test.assign(r=r).groupby("season").r.median().round(2).to_dict())
def mae_k(k, idx): 
    e = (test.margin.values[idx] - test.rating_dif.values[idx] - (k if np.ndim(k)==0 else k[idx])); return np.abs(e).mean()
specs = {"k=1.5":1.5, "k=1.75":1.75, "k=2.0":2.0, "k=2.5":2.5, "k=3.0":3.0, "nfelo hfa_pts":test.hfa_pts.values}
base = 1.5 if False else 2.0
B = 4000; idx_all = np.arange(n)
out = {}
for nm, k in specs.items():
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = mae_k(k, idx) - mae_k(base, idx)
    out[nm] = (mae_k(k, idx_all), mae_k(k, idx_all) - mae_k(base, idx_all), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
print("\nMAE and paired-bootstrap 95%% CI of (MAE_k - MAE_k=%.1f):" % base)
for nm, (mv, d, lo, hi) in out.items():
    print("  %-14s MAE=%.3f  diff=%+.3f  CI[%+.3f, %+.3f]" % (nm, mv, d, lo, hi))
# Market-based sanity check independent of nfelo scale: mean spread_line in home games (2022-25) ~ HFA if schedules balanced
print("\nmarket mean spread_line 2022-25 home games: %.2f ; mean margin: %.2f" % (test.spread_line.mean(), test.margin.mean()))
# what would MAE vs result be if market line were shifted by +delta toward home? (does the market under-price home?)
for dlt in [-0.5, 0, 0.5, 1.0]:
    e = test.margin + test.mkt_spread - dlt
    print("  market line shifted %+.1f toward home: MAE=%.3f bias=%+.3f cover>0 rate=%.3f" % (dlt, e.abs().mean(), e.mean(), (e>0).sum()/(e!=0).sum()))
