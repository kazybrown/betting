"""CRITIC of TT4 (cost of noise reallocations + coherence). Checks: (a) the algebra
home_tt - away_tt = -S and that a +delta/-delta shift equals a spread change of 2*delta;
(b) empirical (not Normal-approx) MAE change from a reallocation x when the true shift is t,
using the actual residual distributions in 2010-2021 and 2022-2025; (c) the noise cost in the
earlier period (is +0.04 specific to 2022-25?); (d) the cost of rounding to .0/.5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import load

g = load(min_season=1999, verbose=False)
print("(a) coherence algebra: max |(home_tt - away_tt) + S| =", float(np.abs((g.home_tt - g.away_tt) + g.S).max()),
      "| after +d/-d shift, implied S' = -(home_tt+d - (away_tt-d)) = S - 2d  -> for d=1.0 the implied spread moves", -2.0, "pts")
rng = np.random.default_rng(1)
for nm, d in [("2010-2021", g[(g.season >= 2010) & (g.season <= 2021)]), ("2022-2025", g[g.test])]:
    e0 = (np.abs(d.r_home).mean() + np.abs(d.r_away).mean()) / 2
    res = []
    for rep in range(200):
        delta = rng.choice([-1, 1], len(d)) * rng.uniform(0.5, 1.5, len(d))
        res.append((np.abs(d.home_score - (d.home_tt + delta)).mean() + np.abs(d.away_score - (d.away_tt - delta)).mean()) / 2 - e0)
    print(f"(c) {nm} n={len(d)}: baseline MAE {e0:.3f}; random realloc U(0.5,1.5) all games dMAE {np.mean(res):+.4f} (sd {np.std(res):.4f})")
    # (b) empirical: residual r (centered at its median so 'true shift' is defined vs the median); shift by t, predict m+x
    r = np.r_[d.r_home.values, d.r_away.values]; r = r - np.median(r)
    print(f"    empirical E|r + t - x| - E|r + t| (x = reallocation applied, t = true shift):")
    print("      x\\t " + "".join(f"{t:>8.2f}" for t in [0, 0.25, 0.5, 1.0, 1.5]))
    for x in [0.5, 1.0, 1.5]:
        print(f"    {x:>5.1f} " + "".join(f"{np.abs(r + t - x).mean() - np.abs(r + t).mean():+8.4f}" for t in [0, 0.25, 0.5, 1.0, 1.5]))
# (d) rounding cost
te = g[g.test]
def rnd(x):  # half-up to .0/.5
    return np.floor(x * 2 + 0.5) / 2
e0 = (np.abs(te.r_home).mean() + np.abs(te.r_away).mean()) / 2
e1 = (np.abs(te.home_score - rnd(te.home_tt)).mean() + np.abs(te.away_score - rnd(te.away_tt)).mean()) / 2
print(f"(d) rounding identity to .0/.5: MAE {e0:.4f} -> {e1:.4f} (d {e1-e0:+.4f}); identity values already on .25 grid: {np.mean((te.home_tt*4).round()==te.home_tt*4):.2f}")
