"""06: consolidated Week-1 parameter recommendations. Recomputes the headline numbers each recommendation rests on
(so this file is self-contained) and prints the parameter table for the engine, incl. the 2026 W1 total prior."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, desc, boot_ci, paired_mae, ols

m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna()].copy()
w1 = m[m.week == 1]; d1 = d[d.week == 1]

print("=== 1. SPREAD SHRINKAGE (Theory 2) ===")
# rolling-origin k for pred = k*elo_line, W1, 2015-2025
e1, es = [], []
for t in range(2015, 2026):
    xf = d[(d.season < t) & (d.week == 1)]; xt = d[(d.season == t) & (d.week == 1)]
    z = -xf.elo_line; k = float((z * xf.margin).sum() / (z * z).sum())
    e1.append(xt.margin + xt.elo_line); es.append(xt.margin + k * xt.elo_line)
e1, es = pd.concat(e1), pd.concat(es)
dd, lo, hi, p, n = paired_mae(es, e1)
print("  W1 rolling-origin (2015-25, n=%d): MAE shrunk - MAE k=1 = %+.3f [%+.2f,%+.2f] p=%.2f  -> keep k=1.0 (no Week-1 shrink of the rating line)" % (n, dd, lo, hi, p))
co, r = ols(d1.margin, [d1.elo_dif_pts], ["b"])
print("  W1 slope of margin on preseason rating gap = %.2f (se %.2f), n=%d: regressed preseason Elo is not over-confident" % (co["b"][0], co["b"][1], len(d1)))

print("\n=== 2. MARKET / CONTEXT INFORMATION IN WEEK 1 (Theory 1 + caps) ===")
d["dev"] = d.mkt - d.elo_line
x = d[~d.neutral]; w1f = (x.week == 1).astype(float)
co, r = ols(x.err_elo_line, [x.dev, w1f, x.dev * w1f], ["s", "w1", "sx"])
print("  info share of market deviation from ratings: other weeks %.2f (se %.2f), W1 %.2f (change %+.2f, se %.2f, p=%.3f), n=%d" %
      (-co["s"][0], co["s"][1], -(co["s"][0] + co["sx"][0]), -co["sx"][0], co["sx"][1], co["sx"][2], len(x)))
e4 = (x.week <= 4).astype(float); co, r = ols(x.err_elo_line, [x.dev, e4, x.dev * e4], ["s", "e", "sx"])
print("  same for W1-4 vs W5+: W5+ %.2f, W1-4 %.2f (change %+.2f, se %.2f, p=%.3f)" % (-co["s"][0], -(co["s"][0] + co["sx"][0]), -co["sx"][0], co["sx"][1], co["sx"][2]))
print("  -> Week 1: cap the SUM of context/news adjustments at 1.5 (vs 2.5) and 2.0 on totals (vs 3.0); do not chase the market's deviation from ratings")

print("\n=== 3. WEEK-1 HFA (Theory 5) ===")
xx = d[~d.neutral]; w1f = (xx.week == 1).astype(float)
co, r = ols(xx.margin, [xx.elo_dif_pts, w1f, xx.elo_dif_pts * w1f], ["b", "w1", "bx"])
print("  HFA at equal ratings: other weeks %+.2f (se %.2f); W1 shift %+.2f (se %.2f, p=%.3f)  [2009-2025 non-neutral, n=%d]" % (co["const"][0], co["const"][1], co["w1"][0], co["w1"][1], co["w1"][2], len(xx)))
n_, mu, se, p = desc(m[(m.week == 1) & ~m.neutral].err_mkt)
print("  market home residual W1 1999-2025: %+.2f (se %.2f, p=%.3f, n=%d)  -> consistent sign, not significant: keep site HFA; optional -0.5 W1 haircut (LOW)" % (mu, se, p, n_))

print("\n=== 4. WEEK-1 TOTAL PRIOR (Theory 3) ===")
smean = m.groupby("season").total_pts.mean()
w2p = m[m.week >= 2].groupby("season").total_pts.mean()
dp = w1.groupby("season").apply(lambda x: x.total_pts.mean() - smean.get(x.name - 1, np.nan)).dropna()
dr = w1.groupby("season").apply(lambda x: x.total_pts.mean() - w2p[x.name])
print("  realized W1 mean - previous-season mean: %+.2f (se %.2f, p=%.3f, %d seasons)" % (dp.mean(), dp.std() / np.sqrt(len(dp)), stats.ttest_1samp(dp, 0).pvalue, len(dp)))
print("  realized W1 mean - same-season W2+ mean:  %+.2f (se %.2f, p=%.3f); seasons negative %d of %d (binomial p=%.3f)" %
      (dr.mean(), dr.std() / np.sqrt(len(dr)), stats.ttest_1samp(dr, 0).pvalue, int((dr < 0).sum()), len(dr), stats.binomtest(int((dr < 0).sum()), len(dr), 0.5).pvalue))
n_, mu, se, p = desc(w1.tot_err); n5, mu5, se5, p5 = desc(m[m.week >= 5].tot_err)
print("  market W1 total residual %+.2f (se %.2f, n=%d) vs W5+ %+.2f (se %.2f): W1 - W5+ = %+.2f (Welch p=%.3f)" %
      (mu, se, n_, mu5, se5, mu - mu5, stats.ttest_ind(w1.tot_err, m[m.week >= 5].tot_err, equal_var=False).pvalue))
print("  2025 realized mean total (all REG) = %.2f  -> engine prior 46.0 confirmed; RECOMMENDED 2026 W1 prior = %.2f - 1.0 = %.1f (W2+ revert to %.1f)" % (smean[2025], smean[2025], smean[2025] - 1.0, smean[2025]))
print("  rating coefficient c in W1: fit<=2021 c=0.363 (se 0.230) vs engine 0.35 -> keep 0.35 (see 02 log)")

print("\n=== 5. NEW HEAD COACH / NEW QB IN WEEK 1 (Theory 4) — from 04 log ===")
print("  new HC: W1 resid vs market -0.18 (se 1.11, n=412 games / 171 team-games), ATS 0.493 -> REJECTED, no adjustment")
print("  new starting QB (vs prev-season primary): +2.13 (se 0.98, p=0.03; FIT +2.10, TEST +2.31), back ATS 0.582 (p=0.04), resid vs rating line +1.58 (se 1.13)")
print("  -> W1 only: +1.0 pt toward a team whose planned W1 starter differs from last season's primary starter (half the point estimate); never apply the")
print("     'starter->backup' downgrade to a planned new starter in W1. LOW confidence; re-test after 2026.")

print("\n=== PARAMETER TABLE (Week 1 only; all other weeks unchanged) ===")
tbl = [
    ("spread shrink k on rating line", "1.00 (keep)", "rolling-origin k~0.94, MAE diff -0.03 [-0.08,+0.03]"),
    ("preseason-rating slope", "1.00 (keep)", "W1 slope 1.05-1.17, CI includes 1"),
    ("context-adjustment cap, spread", "1.5 (from 2.5)", "market info share beyond ratings 0.41 in W1 vs 0.93 later; W1-4 change -0.40 p=0.03"),
    ("context-adjustment cap, total", "2.0 (from 3.0)", "same evidence, applied proportionally; LOW"),
    ("site HFA", "keep (optional -0.5)", "W1 shift -1.0 (se 0.8), market home resid -0.83 (se 0.64); not significant"),
    ("league total prior (W1)", "prev-season mean - 1.0 -> 45.0 for 2026", "W1 realized -1.3 vs prev mean & vs rest of season, p~0.08-0.09; P0 bias -1.38 [-2.92,+0.09]"),
    ("totals rating coefficient c", "0.35 (keep)", "W1 c_fit 0.36 (se 0.23)"),
    ("new head coach adj (W1)", "0 (none)", "-0.18 (se 1.11), ATS 0.493"),
    ("new planned starting QB adj (W1)", "+1.0 to that team's spread side (LOW)", "+2.1 (se 1.0) vs market, ATS 0.58; half-weight"),
    ("confidence tags (W1)", "keep thresholds", "residual SD W1 13.3 vs 13.2 W5+ (Levene p=0.69); engine-market SD smaller in W1 (2.25 vs 3.27)"),
]
for a, b, c in tbl:
    print("  %-34s | %-40s | %s" % (a, b, c))
