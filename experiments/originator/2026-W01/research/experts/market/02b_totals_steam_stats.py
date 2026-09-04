"""02b_totals_steam_stats.py - p-values / CIs for the totals open->close result in 02 (2023-25 only) and
the spread 'steam continues' result by season (stability). Run from research/.
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, ats
m = merged()
t2 = m[m.total_line_open.notna() & m.total_line_close.notna()].copy()
t2["move"] = t2.total_line_close - t2.total_line_open; t2["res"] = t2.total_pts - t2.total_line_close
for lo in [0.5, 1.0]:
    h = t2[t2.move.abs() >= lo]; ws = np.sign(h.move) * np.sign(h.res)
    w, l = int((ws > 0).sum()), int((ws < 0).sum()); bt = stats.binomtest(l, w + l, 0.5)
    ci = bt.proportion_ci(method="wilson")
    print(f"TOTALS fade-steam at close, |move|>={lo}: fade wins {l}-{w} = {l/(w+l):.3f}, 95% CI {ci.low:.3f}-{ci.high:.3f}, binomial p={bt.pvalue:.4f}, n={len(h)}")
    print("   by season:", h.assign(fade=(ws < 0)).groupby("season").fade.agg(["sum", "count"]).T.to_dict())
print("\nSPREAD with-steam ATS at close by season (|move|>=1):")
d = m[m.home_line_open.notna() & m.home_line_close.notna()].copy(); d["move"] = d.home_line_close - d.home_line_open
rows = []
for s, g in d[d.move.abs() >= 1].groupby("season"):
    pred = g.home_line_close + np.sign(g.move) * 0.5; w, l, p = ats(pred, g.home_line_close, g.margin)
    rows.append(dict(season=s, n=len(g), W=w, L=l, P=p, with_steam=w / (w + l)))
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
print("seasons with with-steam > 0.5:", int((r.with_steam > 0.5).sum()), "of", len(r), "| pooled:", round(r.W.sum() / (r.W.sum() + r.L.sum()), 4))
