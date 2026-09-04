"""00: sanity checks on signs, coverage, and the line decomposition used by every later script."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from common import build, mae

pd.set_option("display.width", 220)
m = build(min_season=1999)
print(f"rows (REG, with market line): {len(m)}  seasons {m.season.min()}-{m.season.max()}")
print("sign check corr(mkt, margin) all  = %.3f (must be strongly NEGATIVE)" % np.corrcoef(m.mkt, m.margin)[0, 1])
w1 = m[m.week == 1]
print("sign check corr(mkt, margin) wk1  = %.3f | n=%d" % (np.corrcoef(w1.mkt, w1.margin)[0, 1], len(w1)))
print("sign check corr(mkt_total, total_pts) = %.3f (should be positive)" % np.corrcoef(m.mkt_total, m.total_pts)[0, 1])
print("mean err_mkt (home bias, all) = %+.3f | mean tot_err (all) = %+.3f" % (m.err_mkt.mean(), m.tot_err.mean()))

n9 = m[m.season >= 2009]
have = n9.nfelo_dif_base.notna()
print(f"\n2009+ rows {len(n9)}; with nfelo {int(have.sum())} ({have.mean():.3f}); week-1 with nfelo {int((have & (n9.week==1)).sum())} of {int((n9.week==1).sum())}")
n9 = n9[have]
dec = n9.elo_dif_pts * 25 + n9.hfa_mod + n9.home_net_qb_mod.fillna(0) - n9.nfelo_dif_base
print("decomposition nfelo_dif_base = elo_dif + hfa_mod + qb_mod : |resid|<0.01 in %.3f of rows (max %.2f Elo)" % ((dec.abs() < 0.01).mean(), dec.abs().max()))
print("corr(nraw, mkt) = %.3f | corr(elo_line, mkt) = %.3f | corr(elo_line, margin) = %.3f (negative)" %
      (np.corrcoef(n9.nraw, n9.mkt)[0, 1], np.corrcoef(n9.elo_line, n9.mkt)[0, 1], np.corrcoef(n9.elo_line, n9.margin)[0, 1]))
print("MAE (2009+ REG): market %.3f | nraw %.3f | elo_line (no QB adj) %.3f | elo_only %.3f | nclose %.3f" %
      (mae(-n9.mkt, n9.margin), mae(-n9.nraw, n9.margin), mae(-n9.elo_line, n9.margin), mae(-n9.elo_only, n9.margin), mae(-n9.nclose, n9.margin)))
print("mean hfa_pts by week bucket (nfelo site HFA incl. mods):")
print(n9.groupby("wk").hfa_pts.agg(["mean", "std", "size"]).reindex(["1", "2", "3", "4", "5-9", "10+"]).round(3).to_string())
print("\nWeek-1 starting Elo spread (SD of team ratings, points) by season — how compressed are preseason ratings?")
w1 = n9[n9.week == 1]
sd = w1.groupby("season").apply(lambda d: pd.concat([d.home_rating, d.away_rating]).std()).round(2)
print(sd.to_string())
print("mean W1 rating SD (pts): %.2f ; W10+ rating SD: %.2f" % (sd.mean(), pd.concat([n9[n9.week >= 10].home_rating, n9[n9.week >= 10].away_rating]).std()))
