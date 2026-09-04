"""Cost of applying reallocations that are NOT correlated with truth.
The spec applies +/-0.5..1.5 sum-preserving 'matchup reallocations' to the identity split.
Since no matchup feature tested (05a/05b) predicts the split OOS, the realistic case is that
the reallocation is noise. Simulate on 2022-2025: add delta ~ sign*U(0.5,1.5) to home and
-delta to away (sum preserved) and measure the team-total MAE increase and the implied
spread inconsistency (a sum-preserving reallocation of delta == a spread change of 2*delta).
Also: how large would a real, fully-correct signal need to be to justify a 1.0 reallocation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import load, over_rate

g = load(min_season=1999, verbose=False)
te = g[g.test].copy()
rng = np.random.default_rng(1)
e0 = (np.abs(te.home_score - te.home_tt).mean() + np.abs(te.away_score - te.away_tt).mean()) / 2
print(f"TEST 2022-2025 n={len(te)}: baseline identity team-total MAE (home/away avg) = {e0:.3f}")
for lo_, hi_, share in [(0.5, 1.5, 1.0), (0.5, 1.5, 0.5), (0.5, 1.0, 1.0), (0.5, 0.5, 1.0), (1.5, 1.5, 1.0)]:
    res = []
    for rep in range(200):
        applied = rng.random(len(te)) < share
        delta = rng.choice([-1, 1], len(te)) * rng.uniform(lo_, hi_, len(te)) * applied
        e1 = (np.abs(te.home_score - (te.home_tt + delta)).mean() + np.abs(te.away_score - (te.away_tt - delta)).mean()) / 2
        res.append(e1 - e0)
    print(f"  random realloc U({lo_},{hi_}) applied to {share:.0%} of games: dMAE {np.mean(res):+.4f} (sd over reps {np.std(res):.4f}); implied spread shift 2*delta up to {2*hi_:.1f} pts")

# what does a reallocation of x cost/gain if the true expected shift is t (same sign)?
# E|Y - (m + x)| for Y ~ approx Normal(m + t, sigma=team-score residual SD)
sig = float(np.std(np.r_[te.r_home, te.r_away]))
from scipy.stats import norm
def expabs(mu, s):  # E|N(mu, s)|
    return s * np.sqrt(2/np.pi) * np.exp(-mu**2/(2*s**2)) + mu * (1 - 2*norm.cdf(-mu/s))
print(f"\nresidual SD of team score around identity in test: {sig:.2f}")
print("expected MAE change from reallocating x when the TRUE mean shift is t (same sign), Normal approx:")
print('x\\t  ' + "".join(f"{t:>8.2f}" for t in [0.0, 0.25, 0.5, 1.0, 1.5]))
for x in [0.5, 1.0, 1.5]:
    print(f"{x:>6.1f}" + "".join(f"{expabs(t - x, sig) - expabs(t, sig):+8.4f}" for t in [0.0, 0.25, 0.5, 1.0, 1.5]))
print("(negative = MAE falls; a reallocation of x only pays if the true shift t > x/2)")
