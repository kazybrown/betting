"""CRITIC follow-up on TT1b: the expert says 'at most -0.5 could be justified below 15'. Check the
in-sample (train) and rolling-origin OOS MAE of a -0.5 / -1.0 shade applied only to tt <= 15, and
the rolling P(over) with season-block CI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from common import load, over_rate

g = load(min_season=1999, verbose=False)
rows = []
for side, sc, tt in [("home", "home_score", "home_tt"), ("away", "away_score", "away_tt")]:
    d = g[["gid", "season", "train", "test", sc, tt]].rename(columns={sc: "score", tt: "tt"}); d["side"] = side; rows.append(d)
L = pd.concat(rows, ignore_index=True)
rng = np.random.default_rng(0)
def sb_ci(vals, seasons, n=2000):
    u = np.unique(seasons); grp = {s: vals[seasons == s] for s in u}; bs = []
    for _ in range(n):
        pick = rng.choice(u, len(u), replace=True); bs.append(np.concatenate([grp[s] for s in pick]).mean())
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for thr in [15, 17.5]:
    for nm, d in [("TRAIN 99-21 (in-sample)", L[L.train & (L.tt <= thr)]), ("2009-2025", L[(L.season >= 2009) & (L.tt <= thr)]), ("TEST 22-25", L[L.test & (L.tt <= thr)])]:
        out = []
        for k in [-0.5, -1.0]:
            e0 = np.abs(d.score - d.tt).values; e1 = np.abs(d.score - (d.tt + k)).values; dd = e0 - e1
            lo, hi = sb_ci(dd, d.season.values)
            out.append(f"shade {k}: dMAE {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
        o, n = over_rate(d.score, d.tt)
        print(f"tt<={thr:>4} {nm:>24s} n={len(d):5d}: P(over) {o:.3f} | " + " | ".join(out))
