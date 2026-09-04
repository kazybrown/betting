"""Synthesis check 06: do the general -0.5 level shift (median target) and the Week-1 shift stack?
Base = prior-season realized mean (L_mean) + 0.30*elo_sum in week 1 (no in-season info exists yet)."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research"); sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd
from common import build, paired_mae_ci
m = build(K_team=3, K_lg=128, verbose=False)
r = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna() & (m.season >= 2010)].copy()
r["base"] = r.lg_prev + 0.30 * r.elo_sum
for win, lab in [((2010, 2021), "2010-21"), ((2022, 2025), "2022-25"), ((2010, 2025), "2010-25")]:
    w = r[(r.season >= win[0]) & (r.season <= win[1]) & (r.week == 1)]
    e0 = w.total_pts - w.base
    print(f"[W1 {lab}] n={len(w)} base bias {e0.mean():+.2f} median {e0.median():+.2f} (se {e0.std()/np.sqrt(len(w)):.2f}); market W1 residual {(w.total_pts-w.mkt_total).mean():+.2f}")
    for sh in [-0.5, -0.75, -1.0, -1.25]:
        e = w.total_pts - (w.base + sh); d, lo, hi, _ = paired_mae_ci(e, e0)
        print(f"     shift {sh:+.2f}: bias {e.mean():+.2f} MAE {np.abs(e).mean():.3f} dMAE {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
w2 = r[(r.week >= 2)]
for win, lab in [((2010, 2021), "2010-21"), ((2022, 2025), "2022-25")]:
    w = w2[(w2.season >= win[0]) & (w2.season <= win[1])]
    b = w.lg_blend + 0.30 * w.elo_sum
    e0 = w.total_pts - b
    for sh in [-0.5, -0.75]:
        e = w.total_pts - (b + sh); d, lo, hi, _ = paired_mae_ci(e, e0)
        print(f"[W2+ {lab}] n={len(w)} blend base bias {e0.mean():+.2f} median {e0.median():+.2f}; shift {sh:+.2f}: bias {e.mean():+.2f} dMAE {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
