"""CRITIC 00: raw-data checks before trusting anything.
(a) sign conventions; (b) which 2009+ REG games fail the nfelo join (expert's samples drop them);
(c) wind distribution incl. zeros and by roof; (d) weather coverage by season; (e) roof by season.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd
from common import build
pd.set_option("display.width", 220)
m = build(K_team=1, K_lg=128, verbose=True)
r = m[(m.game_type == "REG")]
print("\n(a) sign checks REG 1999-2025: corr(mkt_spread, margin)=%.3f  corr(mkt_total,total_pts)=%.3f  mean(total-mkt)=%.2f" %
      (np.corrcoef(r.mkt_spread, r.margin)[0, 1], np.corrcoef(r.mkt_total, r.total_pts)[0, 1], (r.total_pts - r.mkt_total).mean()))

print("\n(b) nfelo join gaps, REG 2009+ (games without elo_sum):")
r9 = r[r.season >= 2009]
miss = r9[r9.elo_sum.isna()]
print("   n missing =", len(miss), "of", len(r9), "; by season:", miss.groupby("season").size().to_dict())
print("   by week:", miss.groupby("week").size().to_dict())
print("   teams involved (top):", pd.concat([miss.home, miss.away]).value_counts().head(8).to_dict())
print("   mean total of missing games=%.2f vs present=%.2f ; mean mkt_total missing=%.2f present=%.2f" %
      (miss.total_pts.mean(), r9[r9.elo_sum.notna()].total_pts.mean(), miss.mkt_total.mean(), r9[r9.elo_sum.notna()].mkt_total.mean()))
print("   market MAE on missing games=%.3f vs present=%.3f" % ((miss.total_pts - miss.mkt_total).abs().mean(), (r9[r9.elo_sum.notna()].total_pts - r9[r9.elo_sum.notna()].mkt_total).abs().mean()))
print("   sample of missing gids:", miss.gid.head(10).tolist())

print("\n(c) wind distribution, outdoor REG games with observed wind:")
o = r[(r.outdoor == 1) & r.wind.notna()]
print("   n=%d; wind==0: %d (%.1f%%); wind<=3: %d; quantiles:" % (len(o), (o.wind == 0).sum(), 100 * (o.wind == 0).mean(), (o.wind <= 3).sum()), o.wind.quantile([.1, .25, .5, .75, .9, .95, .99]).round(1).to_dict())
print("   wind==0 share by season:", o.groupby("season").apply(lambda x: round((x.wind == 0).mean(), 2)).to_dict())
print("   wind==0 by roof:", o[o.wind == 0].roof.value_counts().to_dict(), "| temp mean where wind==0: %.1f vs %.1f" % (o[o.wind == 0].temp.mean(), o[o.wind > 0].temp.mean()))
print("   market residual: wind==0 %.2f (n=%d) | 1-3 %.2f (n=%d) | 4-6 %.2f (n=%d) | 7-9 %.2f (n=%d)" % (
    o[o.wind == 0].total_err_mkt.mean(), (o.wind == 0).sum(), o[(o.wind >= 1) & (o.wind <= 3)].total_err_mkt.mean(), ((o.wind >= 1) & (o.wind <= 3)).sum(),
    o[(o.wind >= 4) & (o.wind <= 6)].total_err_mkt.mean(), ((o.wind >= 4) & (o.wind <= 6)).sum(), o[(o.wind >= 7) & (o.wind <= 9)].total_err_mkt.mean(), ((o.wind >= 7) & (o.wind <= 9)).sum()))

print("\n(d) outdoor weather coverage by season (share of outdoor REG games with observed wind AFTER nflfastR fill):")
oo = r[r.outdoor == 1]
print("  ", oo.groupby("season").apply(lambda x: round(x.wind.notna().mean(), 2)).to_dict())
print("   raw nflverse wind coverage (before fill) is what common.py fills; 2022 has no fill source.")
print("\n(e) roof counts by season (REG):")
print(pd.crosstab(r.season, r.roof).to_string())
print("\n(f) 'open' roof = retractable open; is_dome = dome+closed. share of games by roof in test:", r[r.test].roof.value_counts(normalize=True).round(3).to_dict())
