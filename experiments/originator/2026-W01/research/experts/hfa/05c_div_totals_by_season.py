"""THEORY 5 (cont): season-by-season consistency of the divisional totals gap (raw and vs market)."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import merged
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].dropna(subset=["mkt_total"]).copy(); m["div"] = m.div_game.astype(int)
rows = []
for s, d in m.groupby("season"):
    a, b = d[d["div"]==1], d[d["div"]==0]
    rows.append(dict(season=s, n_div=len(a), raw_gap=a.total_pts.mean()-b.total_pts.mean(), mkt_gap=a.mkt_total.mean()-b.mkt_total.mean(),
                     resid_gap=a.total_err_mkt.mean()-b.total_err_mkt.mean(), div_under=(a.total_err_mkt<0).sum()/(a.total_err_mkt!=0).sum()))
T = pd.DataFrame(rows).set_index("season"); print(T.round(2).to_string())
print("seasons with negative resid_gap: %d of %d ; 2022-25: %d of 4" % ((T.resid_gap<0).sum(), len(T), (T.loc[2022:].resid_gap<0).sum()))
print("mean raw gap 2009-25 ex2020 = %.2f ; mean market gap = %.2f ; mean resid gap = %.2f (se across seasons %.2f)" %
      (T[T.index!=2020].raw_gap.mean(), T[T.index!=2020].mkt_gap.mean(), T[T.index!=2020].resid_gap.mean(), T[T.index!=2020].resid_gap.std()/np.sqrt(16)))
