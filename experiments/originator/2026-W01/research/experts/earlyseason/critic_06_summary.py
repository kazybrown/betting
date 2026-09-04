"""CRITIC 06: (i) leakage guard on nfelo hfa_mod for Week 1; (ii) dispersion of the ACTUAL 2026 W1 engine ratings
(PFF / Cole / nfelo) vs the historical nfelo W1 dispersion that the 'k=1.0, no shrink' verdict was validated on;
(iii) consolidated numbers behind each critic verdict (all recomputed here so this file is self-contained)."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, paired_mae, ols, desc
D = "/home/user/originator-2026-w01/research/data/"
m = build(min_season=1999); d = m[m.nfelo_dif_base.notna()].copy()

print("(i) nfelo hfa_base_mod: unique values per season (>2 = rolling/dynamic estimate).  W1 uses only prior-season data by construction.")
n = pd.read_csv(D + "nfelo_games.csv", low_memory=False); n["season"] = n.game_id.str[:4].astype(int); n["week"] = n.game_id.str[5:7].astype(int)
u = n.groupby("season").hfa_base_mod.nunique(); print("  seasons with >2 distinct hfa_base_mod values: %d of %d (first dynamic season %s)" % (int((u > 2).sum()), len(u), u[u > 2].index.min()))
w1n = n[n.week == 1].groupby("season").hfa_base_mod.nunique(); print("  distinct hfa_base_mod values WITHIN week 1 per season: max %d -> W1 HFA is a single pre-season value (no in-season leakage)" % w1n.max())

print("\n(ii) 2026 Week-1 rating dispersion (points vs average) — the shrink verdict (k=1) was validated on nfelo's regressed W1 Elo")
p = pd.read_csv(D + "pff_power_ratings.csv"); c = pd.read_csv(D + "cole_power_rankings.csv")
n26 = n[n.season == 2026]; r26 = pd.concat([(n26.starting_nfelo_home - 1505) / 25, (n26.starting_nfelo_away - 1505) / 25])
w1h = d[d.week == 1]; hist = w1h.groupby("season").apply(lambda x: pd.concat([x.home_rating, x.away_rating]).std())
print("  nfelo W1 rating SD by season 2009-2025: mean %.2f (min %.2f, max %.2f) | 2026 W1 nfelo SD %.2f (n=%d)" % (hist.mean(), hist.min(), hist.max(), r26.std(), len(r26)))
print("  PFF 2026 'Point Spread Rating Points': SD %.2f (range %.1f..%.1f, n=%d) | PFF QB component SD %.2f" % (p["Point Spread Rating Points"].std(), p["Point Spread Rating Points"].min(), p["Point Spread Rating Points"].max(), len(p), p["Point Spread Rating QB"].std()))
print("  Cole 2026 power_ranking SD %.2f | betting_power_ranking SD %.2f (n=%d)" % (c.power_ranking.std(), c.betting_power_ranking.std(), len(c)))
co, r = ols(w1h.margin, [w1h.elo_dif_pts], ["b"])
print("  outcome-optimal dispersion: W1 margin ~ nfelo gap slope %.2f (se %.2f) x nfelo SD %.2f => outcome-supported W1 rating SD ~ %.1f pts (heuristic, 95%% band %.1f-%.1f)" %
      (co["b"][0], co["b"][1], hist.mean(), co["b"][0] * hist.mean(), (co["b"][0] - 1.96 * co["b"][1]) * hist.mean(), (co["b"][0] + 1.96 * co["b"][1]) * hist.mean()))
for lab, sd in [("PFF", p["Point Spread Rating Points"].std()), ("Cole power", c.power_ranking.std()), ("Cole betting", c.betting_power_ranking.std()), ("nfelo 2026", r26.std())]:
    print("    %-12s SD %.2f -> implied W1 shrink to reach ~%.1f: k = %.2f" % (lab, sd, co["b"][0] * hist.mean(), co["b"][0] * hist.mean() / sd))
# 2026 W1 market spread dispersion for comparison
g26 = pd.read_csv(D + "games_1999_2025.csv", low_memory=False); g26 = g26[(g26.season == 2026) & (g26.week == 1)]
print("  2026 W1 market |spread| mean %.2f, SD of spread_line %.2f (historical W1 2009-25 mean |mkt| %.2f)" % (g26.spread_line.abs().mean(), g26.spread_line.std(), w1h.abs_line.mean()))

print("\n(iii) consolidated critic numbers")
x = d[d.week == 1]; dd, lo, hi, pv, nn = paired_mae(x.err_elo_line, x.err_mkt); print("  1a  W1 MAE elo_line %.3f vs market %.3f diff %+.3f [%+.2f,%+.2f] n=%d" % (x.err_elo_line.abs().mean(), x.err_mkt.abs().mean(), dd, lo, hi, nn))
x = d[d.week >= 10]; dd, lo, hi, pv, nn = paired_mae(x.err_elo_line, x.err_mkt); print("  1a  W10+ elo_line - market %+.3f [%+.2f,%+.2f] n=%d (nraw: %+.3f)" % (dd, lo, hi, nn, paired_mae(x.err_nraw, x.err_mkt)[0]))
w1 = m[m.week == 1]; nn_, mu, se, pv = desc(w1.err_mkt); print("  1b  W1 home residual %+.2f (se %.2f, p=%.2f, n=%d); MZ slope %.2f" % (mu, se, pv, nn_, ols(w1.margin, [-w1.mkt], ["b"])[0]["b"][0]))
smean = m.groupby("season").total_pts.mean(); w1 = w1.assign(prev_mean=w1.season.map(lambda s: smean.get(s - 1, np.nan)))
for lab, t0 in [("1999-2025", 2000), ("2009-2025", 2009), ("2015-2025", 2015)]:
    dp = w1[w1.season >= t0].groupby("season").apply(lambda z: z.total_pts.mean() - z.prev_mean.iloc[0]).dropna()
    print("  3   W1 realized - prev-season mean %-9s %+.2f (se %.2f, %d seasons)" % (lab, dp.mean(), dp.std() / np.sqrt(len(dp)), len(dp)))
te = w1[(w1.season >= 2005) & w1.prev_mean.notna()]
for delta in (-0.5, -0.75, -1.0):
    e = te.total_pts - (te.prev_mean + delta); e0 = te.total_pts - te.prev_mean; dd, lo, hi, pv, nn = paired_mae(e, e0)
    print("  3   fixed W1 shift %+.2f, 2005-25 n=%d: MAE diff vs prev-mean prior %+.3f [%+.2f,%+.2f] p=%.2f, bias %+.2f" % (delta, nn, dd, lo, hi, pv, e.mean()))
print("  3   2025 realized mean total %.2f -> revised 2026 W1 prior = %.2f - 0.75 = %.2f" % (smean[2025], smean[2025], smean[2025] - 0.75))
