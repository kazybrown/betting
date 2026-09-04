"""CRITIC 03 / Theory 3: Week-1 totals level.  The expert recommends W1 prior = prev-season mean - 1.0.
Attacks:
  A. Reproduce: W1 market residual, W1 realized vs prev-season mean and vs same-season W2+ mean, 20/27 sign test.
  B. Is the dip W1-SPECIFIC?  Realized total by week (season-demeaned) for weeks 1..8 with season-clustered SEs;
     and the market's own week profile.  If W2/W3 are equally low, the 'W1 rust' story is wrong (it would be a
     September effect, and the prior change should not revert in W2).
  C. Era stability: 1999-2008 / 2009-2016 / 2017-2025 for (W1 - prev mean) and (W1 - W2+ mean); 2009+ and 2015+ only.
  D. What does the -1.0 shift actually do OOS?  Rolling-origin 2005-25 and 2015-25: P0 + delta for delta in
     {0,-0.5,-1.0,-1.5,-2.0} -> MAE, bias, and paired MAE vs P0 and vs the MARKET; plus the engine's actual
     implied total (prior + 0.35*rating_sum, nfelo path, 2009+) with and without the shift vs the market.
  E. Placebo: apply the same 'previous-season-mean minus delta' logic to Week 2 and Week 3 (should show ~0 if W1-specific).
  F. Median / trimmed versions (the W1 residual distribution is skewed: median -2.0 vs mean -0.47).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, desc, boot_ci, binom, paired_mae, ols

pd.set_option("display.width", 250)
m = build(min_season=1999)
smean = m.groupby("season").total_pts.mean(); smean_w2p = m[m.week >= 2].groupby("season").total_pts.mean()
m["prev_mean"] = m.season.map(lambda s: smean.get(s - 1, np.nan)); m["cur_w2p"] = m.season.map(smean_w2p); m["cur_mean"] = m.season.map(smean)
w1 = m[m.week == 1].copy()
print("sanity corr(mkt_total, total_pts) = %.3f (positive)" % np.corrcoef(m.mkt_total, m.total_pts)[0, 1])

print("\nA. Reproduce (1999-2025)")
n, mu, se, p = desc(w1.tot_err); lo, hi = boot_ci(w1.tot_err)
u = int((w1.tot_err < 0).sum()); o = int((w1.tot_err > 0).sum()); pct, plo, phi, pb = binom(u, o)
print("  W1 market residual %+.2f (se %.2f) [%+.2f,%+.2f] p=%.3f n=%d | under %d-%d %.3f [%.3f,%.3f] p=%.3f" % (mu, se, lo, hi, p, n, u, o, pct, plo, phi, pb))
dp = w1.groupby("season").apply(lambda x: x.total_pts.mean() - x.prev_mean.iloc[0]).dropna()
dr = w1.groupby("season").apply(lambda x: x.total_pts.mean() - x.cur_w2p.iloc[0])
print("  W1 realized - prev-season mean %+.2f (se %.2f, p=%.3f, %d seasons) | W1 realized - same-season W2+ mean %+.2f (se %.2f, p=%.3f); negative %d/%d binom p=%.3f" %
      (dp.mean(), dp.std() / np.sqrt(len(dp)), stats.ttest_1samp(dp, 0).pvalue, len(dp), dr.mean(), dr.std() / np.sqrt(len(dr)), stats.ttest_1samp(dr, 0).pvalue, int((dr < 0).sum()), len(dr), stats.binomtest(int((dr < 0).sum()), len(dr), 0.5).pvalue))
print("  Wilcoxon signed-rank on (W1 - W2+ mean) per season: p=%.3f | median %+.2f" % (stats.wilcoxon(dr).pvalue, dr.median()))

print("\nB. Week profile of scoring within a season (realized total minus own-season mean), season-clustered SE; and the market total minus season mean")
m["dm_real"] = m.total_pts - m.cur_mean; m["dm_mkt"] = m.mkt_total - m.cur_mean
for wk in list(range(1, 9)) + ["9-13", "14+"]:
    mask = (m.week == wk) if isinstance(wk, int) else (m.week.between(9, 13) if wk == "9-13" else (m.week >= 14))
    x = m[mask]
    r = sm.OLS(x.dm_real.values, np.ones(len(x))).fit(cov_type="cluster", cov_kwds={"groups": x.season.values})
    r2 = sm.OLS(x.dm_mkt.values, np.ones(len(x))).fit(cov_type="cluster", cov_kwds={"groups": x.season.values})
    print("  wk %-5s n=%4d realized - season mean %+.2f (cl se %.2f, p=%.3f) | market - season mean %+.2f (se %.2f) | market resid %+.2f" %
          (str(wk), len(x), r.params[0], r.bse[0], r.pvalues[0], r2.params[0], r2.bse[0], x.tot_err.mean()))
# W1 vs W2-4 directly (same September weather)
x = m[m.week <= 4].copy(); x["w1"] = (x.week == 1).astype(float)
r = sm.OLS(x.dm_real.values, sm.add_constant(x.w1.values)).fit(cov_type="cluster", cov_kwds={"groups": x.season.values})
print("  W1 minus W2-4 (realized, season-demeaned): %+.2f (cl se %.2f, p=%.3f)" % (r.params[1], r.bse[1], r.pvalues[1]))
r = sm.OLS(x.tot_err.values, sm.add_constant(x.w1.values)).fit(cov_type="cluster", cov_kwds={"groups": x.season.values})
print("  W1 minus W2-4 (market residual):            %+.2f (cl se %.2f, p=%.3f)" % (r.params[1], r.bse[1], r.pvalues[1]))

print("\nC. Era stability")
for lab, ss in [("1999-2008", (1999, 2008)), ("2009-2016", (2009, 2016)), ("2017-2025", (2017, 2025)), ("2009-2025", (2009, 2025)), ("2015-2025", (2015, 2025)), ("2022-2025", (2022, 2025))]:
    a_ = dp[(dp.index >= ss[0]) & (dp.index <= ss[1])]; b_ = dr[(dr.index >= ss[0]) & (dr.index <= ss[1])]
    x = w1[w1.season.between(*ss)]
    print("  %-9s W1 - prev mean %+.2f (se %.2f, n_seasons %d) | W1 - W2+ mean %+.2f (se %.2f) neg %d/%d | market resid %+.2f (se %.2f) | market - prev mean %+.2f" %
          (lab, a_.mean(), a_.std() / np.sqrt(len(a_)), len(a_), b_.mean(), b_.std() / np.sqrt(len(b_)), int((b_ < 0).sum()), len(b_), x.tot_err.mean(), x.tot_err.std() / np.sqrt(len(x)), (x.mkt_total - x.prev_mean).mean()))

print("\nD. Rolling-origin effect of a fixed W1 shift: prior = prev-season mean + delta (no fitting needed); pooled W1 games")
for lab, t0 in [("2005-2025", 2005), ("2015-2025", 2015), ("2022-2025", 2022)]:
    te = w1[(w1.season >= t0) & w1.prev_mean.notna()]
    print("  %s n=%d | market MAE %.3f bias %+.2f" % (lab, len(te), te.tot_err.abs().mean(), te.tot_err.mean()))
    e0 = te.total_pts - te.prev_mean
    for delta in [0.0, -0.5, -0.6, -1.0, -1.3, -1.5, -2.0]:
        e = te.total_pts - (te.prev_mean + delta); dd, lo, hi, p, n = paired_mae(e, e0); dm, lom, him, pm, _ = paired_mae(e, te.tot_err)
        print("    delta %+.1f  MAE %.3f bias %+.2f | vs P0: %+.3f [%+.2f,%+.2f] p=%.2f | vs market: %+.3f [%+.2f,%+.2f] p=%.2f" % (delta, e.abs().mean(), e.mean(), dd, lo, hi, p, dm, lom, him, pm))
# engine path with ratings (2009+): prior + 0.35*(rating_sum)
d = m[m.nfelo_dif_base.notna() & (m.week == 1) & m.prev_mean.notna()].copy(); d["rs"] = d.home_rating + d.away_rating
print("  Engine implied total (nfelo path, prior + 0.35*rating_sum), W1 2009-2025 n=%d: mean rating_sum %+.2f" % (len(d), d.rs.mean()))
e0 = d.total_pts - (d.prev_mean + 0.35 * d.rs)
for delta in [0.0, -0.5, -1.0, -1.5]:
    e = d.total_pts - (d.prev_mean + delta + 0.35 * d.rs); dd, lo, hi, p, n = paired_mae(e, e0); dm, lom, him, pm, _ = paired_mae(e, d.tot_err)
    print("    delta %+.1f  MAE %.3f bias %+.2f | vs engine delta 0: %+.3f [%+.2f,%+.2f] p=%.2f | vs market (MAE %.3f): %+.3f [%+.2f,%+.2f] p=%.2f" % (delta, e.abs().mean(), e.mean(), dd, lo, hi, p, d.tot_err.abs().mean(), dm, lom, him, pm))
d15 = d[d.season >= 2015]; e0 = d15.total_pts - (d15.prev_mean + 0.35 * d15.rs)
for delta in [0.0, -1.0]:
    e = d15.total_pts - (d15.prev_mean + delta + 0.35 * d15.rs); dd, lo, hi, p, n = paired_mae(e, e0)
    print("    2015-25 n=%d delta %+.1f MAE %.3f bias %+.2f | vs delta 0: %+.3f [%+.2f,%+.2f] p=%.2f" % (n, delta, e.abs().mean(), e.mean(), dd, lo, hi, p))

print("\nE. Placebo: same shift logic applied to Week 2 and Week 3 (prior = prev-season mean + delta)")
for wk in (2, 3, 4):
    te = m[(m.week == wk) & (m.season >= 2005) & m.prev_mean.notna()]
    e0 = te.total_pts - te.prev_mean; e = te.total_pts - (te.prev_mean - 1.0); dd, lo, hi, p, n = paired_mae(e, e0)
    print("  W%d n=%d bias of prev-mean prior %+.2f (se %.2f) | delta -1.0 MAE diff vs P0 %+.3f [%+.2f,%+.2f] p=%.2f | market resid %+.2f" % (wk, n, e0.mean(), e0.std() / np.sqrt(n), dd, lo, hi, p, te.tot_err.mean()))

print("\nF. Robust location of the W1 market residual and of (W1 - prev mean) game-level")
x = w1.tot_err
print("  W1 market residual: mean %+.2f median %+.2f 10%%-trimmed mean %+.2f | Wilcoxon p=%.3f" % (x.mean(), x.median(), stats.trim_mean(x, 0.1), stats.wilcoxon(x[x != 0]).pvalue))
y = (w1.total_pts - w1.prev_mean).dropna()
print("  W1 realized - prev mean (game level, n=%d): mean %+.2f median %+.2f trimmed %+.2f | t p=%.3f, Wilcoxon p=%.3f" % (len(y), y.mean(), y.median(), stats.trim_mean(y, 0.1), stats.ttest_1samp(y, 0).pvalue, stats.wilcoxon(y).pvalue))
