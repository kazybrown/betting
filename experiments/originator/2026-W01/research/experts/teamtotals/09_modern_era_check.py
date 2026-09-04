"""Robustness for Theory 1: repeat the identity-slope test on the modern era only
(fit 2009-2021, test 2022-2025) so the conclusion does not depend on 1999-2008 lines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from common import load, boot_ci

g = load(min_season=2009, verbose=False)
tr, te = g[g.train], g[g.test]
print(f"modern era: train 2009-2021 n={len(tr)}, test 2022-2025 n={len(te)}")
for y, hyp in [("home_score", "S = -0.5, T = 0.5"), ("away_score", "S = 0.5, T = 0.5"), ("fav_score", "abs_S = 0.5, T = 0.5"), ("dog_score", "abs_S = -0.5, T = 0.5")]:
    X = ["abs_S", "T"] if hyp.startswith("abs_S") else ["S", "T"]
    r = sm.OLS(tr[y], sm.add_constant(tr[X])).fit(cov_type="HC1")
    w = r.wald_test(hyp, scalar=True)
    print(f"  {y:>10s}: " + ", ".join(f"{k} {r.params[k]:+.3f} (se {r.bse[k]:.3f})" for k in r.params.index) + f" | Wald slopes=identity p={float(w.pvalue):.3f}")
    p_lin = r.predict(sm.add_constant(te[X]))
    tt = {"home_score": te.home_tt, "away_score": te.away_tt, "fav_score": te.fav_tt, "dog_score": te.dog_tt}[y]
    e0 = np.abs(te[y] - tt).values; e1 = np.abs(te[y] - p_lin).values
    lo, hi = boot_ci(e0 - e1)
    print(f"              OOS MAE identity {e0.mean():.3f} vs OLS {e1.mean():.3f}: dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}]")
