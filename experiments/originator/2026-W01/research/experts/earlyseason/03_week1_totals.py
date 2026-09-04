"""03 / THEORY 3: Week-1 totals — realized vs market by season; systematic under/over?  And what league prior
should the engine use in Week 1 (it currently uses last season's realized mean, 46.0 for 2025 -> 2026)?
Uses 1999-2025 (market totals available every season).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, desc, boot_ci, binom, paired_mae, ols

pd.set_option("display.width", 250)
m = build(min_season=1999)
smean = m.groupby("season").total_pts.mean()                     # realized season mean (all REG)
smean_w2p = m[m.week >= 2].groupby("season").total_pts.mean()    # season mean excluding week 1
m["prev_mean"] = m.season.map(lambda s: smean.get(s - 1, np.nan))
m["cur_mean"] = m.season.map(smean)
m["cur_mean_w2p"] = m.season.map(smean_w2p)
w1 = m[m.week == 1].copy()

print("A. Week-1 totals by season (1999-2025): market vs realized; O-U record; realized W1 vs previous-season mean and vs rest of this season")
print("  season  n | mkt_total realized  resid | O-U   | prev-season mean | W1 real - prev | W1 mkt - prev | W1 real - rest-of-season")
neg_seasons = 0
for s, x in w1.groupby("season"):
    o = int((x.tot_err > 0).sum()); u = int((x.tot_err < 0).sum())
    r = x.tot_err.mean(); neg_seasons += r < 0
    print("  %d   %2d | %6.2f   %6.2f   %+6.2f | %2d-%2d | %6.2f | %+6.2f | %+6.2f | %+6.2f" %
          (s, len(x), x.mkt_total.mean(), x.total_pts.mean(), r, o, u, x.prev_mean.mean(), x.total_pts.mean() - x.prev_mean.mean(),
           x.mkt_total.mean() - x.prev_mean.mean(), x.total_pts.mean() - x.cur_mean_w2p.mean()))
print("  seasons with W1 realized < market: %d of %d (binomial p=%.3f)" % (neg_seasons, w1.season.nunique(), stats.binomtest(int(neg_seasons), int(w1.season.nunique()), 0.5).pvalue))

print("\nB. Pooled Week-1 total residual (realized - market) with CIs; W2-4 and W5+ controls; and W1 minus W5+ difference")
for lab, x in [("W1 ALL 1999-2025", w1), ("W1 FIT 1999-2021", w1[w1.fit]), ("W1 TEST 2022-2025", w1[w1.test]), ("W1 2009-2025", w1[w1.season >= 2009]),
               ("W1 2015-2025", w1[w1.season >= 2015]), ("W2-4 ALL", m[m.week.between(2, 4)]), ("W5+ ALL", m[m.week >= 5]), ("W5+ TEST", m[(m.week >= 5) & m.test])]:
    n, mu, se, p = desc(x.tot_err); lo, hi = boot_ci(x.tot_err)
    o = int((x.tot_err > 0).sum()); u = int((x.tot_err < 0).sum()); pct, plo, phi, pb = binom(o, u)
    print("  %-20s n=%4d resid %+.2f (se %.2f) 95%%CI [%+.2f,%+.2f] p=%.3f | over %d-%d %.3f [%.3f,%.3f] p=%.3f | median resid %+.1f" %
          (lab, n, mu, se, lo, hi, p, o, u, pct, plo, phi, pb, x.tot_err.median()))
a, b = w1.tot_err, m[m.week >= 5].tot_err
print("  W1 minus W5+ residual: %+.2f  Welch p=%.3f | W1 minus W2-4: %+.2f p=%.3f" %
      (a.mean() - b.mean(), stats.ttest_ind(a, b, equal_var=False).pvalue, a.mean() - m[m.week.between(2, 4)].tot_err.mean(),
       stats.ttest_ind(a, m[m.week.between(2, 4)].tot_err, equal_var=False).pvalue))
# per-season cluster-robust: regress tot_err on W1 dummy with season clusters
import statsmodels.api as sm
X = sm.add_constant((m.week == 1).astype(float).values)
r = sm.OLS(m.tot_err.values, X).fit(cov_type="cluster", cov_kwds={"groups": m.season.values})
print("  cluster(season)-robust: W1 dummy coef %+.2f (se %.2f, p=%.3f); base (W2+) %+.2f" % (r.params[1], r.bse[1], r.pvalues[1], r.params[0]))

print("\nC. Is Week 1 lower-scoring than the rest of its own season?  (realized W1 mean - realized W2+ mean, per season)")
diff = w1.groupby("season").apply(lambda x: x.total_pts.mean() - x.cur_mean_w2p.iloc[0])
print("  mean %+.2f (se %.2f) p=%.3f over %d seasons; seasons negative: %d" % (diff.mean(), diff.std() / np.sqrt(len(diff)), stats.ttest_1samp(diff, 0).pvalue, len(diff), int((diff < 0).sum())))
diff_mkt = w1.groupby("season").apply(lambda x: x.mkt_total.mean() - x.cur_mean_w2p.iloc[0])
print("  market W1 total - realized W2+ mean: %+.2f (se %.2f) -> the market %s the season's scoring level in W1" % (diff_mkt.mean(), diff_mkt.std() / np.sqrt(len(diff_mkt)), "overshoots" if diff_mkt.mean() > 0 else "undershoots"))
diff_prev = w1.groupby("season").apply(lambda x: x.total_pts.mean() - x.prev_mean.iloc[0]).dropna()
diff_prev_mkt = w1.groupby("season").apply(lambda x: x.mkt_total.mean() - x.prev_mean.iloc[0]).dropna()
print("  realized W1 - previous-season mean: %+.2f (se %.2f) p=%.3f | market W1 - previous-season mean: %+.2f (se %.2f)" %
      (diff_prev.mean(), diff_prev.std() / np.sqrt(len(diff_prev)), stats.ttest_1samp(diff_prev, 0).pvalue, diff_prev_mkt.mean(), diff_prev_mkt.std() / np.sqrt(len(diff_prev_mkt))))
print("  same, 2009-2025 only: realized W1 - prev mean %+.2f (se %.2f) | market W1 - prev mean %+.2f" %
      (diff_prev[diff_prev.index >= 2009].mean(), diff_prev[diff_prev.index >= 2009].std() / np.sqrt((diff_prev.index >= 2009).sum()), diff_prev_mkt[diff_prev_mkt.index >= 2009].mean()))

print("\nD. What Week-1 league prior minimizes OOS error?  Candidates evaluated rolling-origin (delta fitted on seasons < t), t = 2005..2025")
print("   P0 = previous-season mean (engine); P1 = P0 + delta_fit; P2 = mean of previous 3 seasons; P3 = P0 + trend (mean of last 3 yr-over-yr changes);")
print("   M  = market W1 total; M1 = market + delta_fit(market resid)")
cands = {k: [] for k in ["P0", "P1", "P2", "P3", "M", "M1"]}; actual = []
for t in range(2005, 2026):
    tr = w1[(w1.season < t) & w1.prev_mean.notna()]; te = w1[w1.season == t]
    if len(te) == 0: continue
    delta = (tr.total_pts - tr.prev_mean).mean()            # avg W1 realized minus prev-season mean, on prior seasons
    dm = tr.tot_err.mean()                                  # avg market residual in prior W1s
    p2 = np.mean([smean[t - k] for k in (1, 2, 3)])
    p3 = smean[t - 1] + np.mean([smean[t - k] - smean[t - k - 1] for k in (1, 2, 3)])
    cands["P0"].append(te.prev_mean); cands["P1"].append(te.prev_mean + delta); cands["P2"].append(pd.Series(p2, index=te.index))
    cands["P3"].append(pd.Series(p3, index=te.index)); cands["M"].append(te.mkt_total); cands["M1"].append(te.mkt_total + dm); actual.append(te.total_pts)
actual = pd.concat(actual)
res = {k: pd.concat(v) for k, v in cands.items()}
print("  pooled OOS W1 games n=%d (2005-2025)" % len(actual))
for k, v in res.items():
    e = actual - v; lo, hi = boot_ci(e)
    print("    %-3s MAE %.3f RMSE %.3f bias %+.2f [%+.2f,%+.2f]" % (k, e.abs().mean(), np.sqrt((e ** 2).mean()), e.mean(), lo, hi))
for a_, b_ in [("P1", "P0"), ("P2", "P0"), ("M", "P0"), ("M1", "M")]:
    dd, lo, hi, p, n = paired_mae(actual - res[a_], actual - res[b_])
    print("    MAE %s - %s = %+.3f [%+.2f,%+.2f] p=%.3f" % (a_, b_, dd, lo, hi, p))
te = w1[w1.test]
print("  2022-2025 only (n=%d): " % len(te) + " | ".join("%s MAE %.2f bias %+.2f" % (k, (actual.loc[te.index] - v.loc[te.index]).abs().mean(), (actual.loc[te.index] - v.loc[te.index]).mean()) for k, v in res.items()))
# delta history: what would the fitted delta have been each year (stability)?
print("  fitted delta (W1 realized - prev-season mean, seasons < t): " + ", ".join("%d:%+.1f" % (t, (w1[(w1.season < t) & w1.prev_mean.notna()].total_pts - w1[(w1.season < t) & w1.prev_mean.notna()].prev_mean).mean()) for t in (2010, 2015, 2020, 2022, 2024, 2026)))

print("\nE. Week-1 totals: dome vs outdoors, hot-weather, and by market total level (is the bias concentrated?)")
for lab, mask in [("dome/closed", w1.is_dome), ("outdoors", ~w1.is_dome), ("outdoors temp>=80F", (~w1.is_dome) & (w1.temp >= 80)), ("outdoors temp<80F", (~w1.is_dome) & (w1.temp < 80)),
                  ("mkt_total < 42", w1.mkt_total < 42), ("mkt_total 42-47.5", w1.mkt_total.between(42, 47.5)), ("mkt_total >= 48", w1.mkt_total >= 48),
                  ("W1 primetime (kick>=20:00 ET)", w1.gametime.notna() & (pd.to_numeric(w1.gametime.str.slice(0, 2), errors="coerce") >= 20)),
                  ("W1 non-primetime (gametime known)", w1.gametime.notna() & (pd.to_numeric(w1.gametime.str.slice(0, 2), errors="coerce") < 20))]:
    x = w1[mask.fillna(False)]; n, mu, se, p = desc(x.tot_err); o = int((x.tot_err > 0).sum()); u = int((x.tot_err < 0).sum())
    print("  %-34s n=%3d resid %+.2f (se %.2f) p=%.2f | over %d-%d %.3f" % (lab, n, mu, se, p, o, u, o / max(o + u, 1)))
# control: same splits weeks 5+
print("  control W5+: dome resid %+.2f (n=%d) | outdoors %+.2f (n=%d)" % (m[(m.week >= 5) & m.is_dome].tot_err.mean(), int(((m.week >= 5) & m.is_dome).sum()), m[(m.week >= 5) & ~m.is_dome].tot_err.mean(), int(((m.week >= 5) & ~m.is_dome).sum())))

print("\nF. Exploitability check: 'under every Week-1 total' ATS by era, and W1 under at -110 break-even 52.4%")
for lab, x in [("1999-2008", w1[w1.season <= 2008]), ("2009-2016", w1[w1.season.between(2009, 2016)]), ("2017-2025", w1[w1.season >= 2017]), ("ALL", w1)]:
    u = int((x.tot_err < 0).sum()); o = int((x.tot_err > 0).sum()); pct, lo, hi, p = binom(u, o)
    print("  %-9s under %d-%d = %.3f [%.3f,%.3f] p=%.3f" % (lab, u, o, pct, lo, hi, p))
