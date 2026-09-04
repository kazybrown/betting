"""01 / THEORY 1: how predictive are (regressed) preseason ratings in weeks 1-4 vs 5+, and is the
Week-1 market less efficient?
  A. by week bucket: MAE of market vs nfelo raw / rating-only lines; OLS slope of margin on each line
     (slope 1 = calibrated; <1 = line should be shrunk; >1 = expanded), R^2.  ALL / FIT / TEST.
  B. 'who is right when they disagree': margin ~ rating line + market line by bucket (FIT and TEST).
  C. Week-1 market biases 1999-2025 (n~430): home bias, favourite cover by size, dog ATS, Mincer-
     Zarnowitz slope, totals over/under — each vs weeks 5+ as control, with CIs / p-values.
  D. week-by-week appendix table.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, desc, boot_ci, paired_mae, binom, ols, BUCKETS

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna()].copy()          # 2009+ with ratings


def slope(y, line):
    """OLS margin ~ -line: returns slope, se, R^2."""
    co, r = ols(y, [-line], ["b"])
    return co["b"][0], co["b"][1], r.rsquared


def block_A(title, dd):
    print(f"\nA. {title}")
    print("  wk    n | MAE  mkt   nraw  elo_line elo_only | nraw-mkt (95%CI)      | slope mkt (se)  R2   | slope nraw (se) R2   | slope elo_line (se) R2 | slope elo_only (se) R2")
    for w in BUCKETS:
        x = dd[dd.wk == w]
        if len(x) < 30: continue
        dif, lo, hi, p, n = paired_mae(x.err_nraw, x.err_mkt)
        sm_, sem, r2m = slope(x.margin, x.mkt); sn, sen, r2n = slope(x.margin, x.nraw)
        se_, see, r2e = slope(x.margin, x.elo_line); so, seo, r2o = slope(x.margin, x.elo_only)
        print("  %-4s %4d | %.2f %.2f %.2f  %.2f     | %+.3f [%+.2f,%+.2f] p=%.2f | %.2f (%.2f) %.3f | %.2f (%.2f) %.3f | %.2f (%.2f) %.3f     | %.2f (%.2f) %.3f" %
              (w, len(x), x.err_mkt.abs().mean(), x.err_nraw.abs().mean(), x.err_elo_line.abs().mean(), x.err_elo_only.abs().mean(),
               dif, lo, hi, p, sm_, sem, r2m, sn, sen, r2n, se_, see, r2e, so, seo, r2o))


block_A("ALL 2009-2025 (REG)", d)
block_A("FIT 2009-2021", d[d.fit])
block_A("TEST 2022-2025", d[d.test])

# pooled early (1-4) vs late (5+) comparison of rating-line slope with a formal interaction test
print("\nA2. Is the rating line (elo_line, no QB adj) less/more predictive early? OLS margin ~ -elo_line * early, HC1")
for lab, dd in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test])]:
    early = (dd.week <= 4).astype(float)
    co, r = ols(dd.margin, [-dd.elo_line, early, -dd.elo_line * early], ["b", "early", "b_x_early"])
    print("  %-4s n=%4d slope late=%.3f (se %.3f)  early-late diff=%+.3f (se %.3f, p=%.3f)  early intercept shift %+.2f (se %.2f)" %
          (lab, len(dd), co["b"][0], co["b"][1], co["b_x_early"][0], co["b_x_early"][1], co["b_x_early"][2], co["early"][0], co["early"][1]))
    co, r = ols(dd.margin, [-dd.mkt, early, -dd.mkt * early], ["b", "early", "b_x_early"])
    print("       market: slope late=%.3f (se %.3f)  early-late diff=%+.3f (se %.3f, p=%.3f)" % (co["b"][0], co["b"][1], co["b_x_early"][0], co["b_x_early"][1], co["b_x_early"][2]))
    w1 = (dd.week == 1).astype(float)
    co, r = ols(dd.margin, [-dd.elo_line, w1, -dd.elo_line * w1], ["b", "w1", "b_x_w1"])
    print("       elo_line week1 only: slope other=%.3f  W1-other diff=%+.3f (se %.3f, p=%.3f)" % (co["b"][0], co["b_x_w1"][0], co["b_x_w1"][1], co["b_x_w1"][2]))

print("\nB. Who is right when rating and market disagree: OLS margin ~ -elo_line, -mkt (weights sum ~1 if both calibrated)")
print("  %-5s %-4s %4s | w_rating (se) | w_market (se) | R2   | MAE of fitted combo (in-bucket, in-sample)" % ("split", "wk", "n"))
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    for w in ["1", "2", "3", "4", "5-9", "10+"]:
        x = dd[dd.wk == w]
        co, r = ols(x.margin, [-x.elo_line, -x.mkt], ["rating", "market"])
        print("  %-5s %-4s %4d | %+.2f (%.2f)  | %+.2f (%.2f)  | %.3f | %.2f" % (lab, w, len(x), co["rating"][0], co["rating"][1], co["market"][0], co["market"][1], r.rsquared, np.abs(r.resid).mean()))
# OOS version: fit combo weights on FIT by bucket, apply to TEST, compare MAE with market alone
print("  OOS: weights fitted on FIT (<=2021) per bucket, applied to TEST 2022-2025")
for w in ["1", "2", "3", "4", "5-9", "10+"]:
    xf = d[d.fit & (d.wk == w)]; xt = d[d.test & (d.wk == w)]
    co, r = ols(xf.margin, [-xf.elo_line, -xf.mkt], ["rating", "market"])
    pred = co["const"][0] + co["rating"][0] * (-xt.elo_line) + co["market"][0] * (-xt.mkt)
    dif, lo, hi, p, n = paired_mae(xt.margin - pred, xt.err_mkt)
    print("    wk %-4s n_test=%3d  w_rating=%+.2f w_market=%+.2f -> MAE combo-mkt %+.3f [%+.2f,%+.2f] p=%.2f" % (w, len(xt), co["rating"][0], co["market"][0], dif, lo, hi, p))

# ---------------- C. Week-1 market bias tests, 1999-2025 ----------------
print("\nC. Week-1 market biases 1999-2025 (n=%d W1 games) vs weeks 5+ control" % (m.week == 1).sum())
def bias_row(lab, x):
    n, mu, se, p = desc(x.err_mkt)
    lo, hi = boot_ci(x.err_mkt)
    f = x[x.fav_sgn != 0]
    fc = f.err_mkt * f.fav_sgn
    nf, muf, sef, pf = desc(fc)
    # ATS of the DOG: dog covers when fav_cover < 0
    dw = int((fc < 0).sum()); dl = int((fc > 0).sum()); dp = int((fc == 0).sum())
    pct, plo, phi, pb = binom(dw, dl)
    # home ATS
    hw = int((x.err_mkt > 0).sum()); hl = int((x.err_mkt < 0).sum())
    hpct, hlo, hhi, hpb = binom(hw, hl)
    # MZ slope
    b, seb, r2 = slope(x.margin, x.mkt)
    tn, tmu, tse, tp = desc(x.tot_err)
    ov = int((x.tot_err > 0).sum()); un = int((x.tot_err < 0).sum())
    opct, olo, ohi, opb = binom(ov, un)
    print("  %-22s n=%4d | home bias %+.2f [%+.2f,%+.2f] p=%.2f | home ATS %.3f [%.2f,%.2f] | fav cover %+.2f (se %.2f) p=%.2f | dog ATS %d-%d-%d %.3f [%.2f,%.2f] p=%.2f | MZ slope %.2f (se %.2f) | tot resid %+.2f (se %.2f) p=%.2f | over %.3f [%.2f,%.2f] p=%.2f" %
          (lab, n, mu, lo, hi, p, hpct, hlo, hhi, muf, sef, pf, dw, dl, dp, pct, plo, phi, pb, b, seb, tmu, tse, tp, opct, olo, ohi, opb))

for lab, mask in [("W1 ALL 1999-2025", m.week == 1), ("W1 FIT 1999-2021", (m.week == 1) & m.fit), ("W1 TEST 2022-2025", (m.week == 1) & m.test),
                  ("W1 1999-2008", (m.week == 1) & (m.season <= 2008)), ("W1 2009-2016", (m.week == 1) & m.season.between(2009, 2016)), ("W1 2017-2025", (m.week == 1) & (m.season >= 2017)),
                  ("W2-4 ALL", m.week.between(2, 4)), ("W5+ ALL (control)", m.week >= 5), ("W5+ TEST", (m.week >= 5) & m.test)]:
    bias_row(lab, m[mask])

print("\n  Week-1 favourite cover margin by line size (1999-2025), + = favourite covered by this much; and dog ATS")
w1 = m[(m.week == 1) & (m.fav_sgn != 0)]
ctrl = m[(m.week >= 5) & (m.fav_sgn != 0)]
for lo_, hi_ in [(0.5, 3.0), (3.5, 6.5), (7.0, 9.5), (10.0, 30.0)]:
    x = w1[w1.abs_line.between(lo_, hi_)]; c = ctrl[ctrl.abs_line.between(lo_, hi_)]
    fc = x.err_mkt * x.fav_sgn; fcc = c.err_mkt * c.fav_sgn
    n, mu, se, p = desc(fc); dw = int((fc < 0).sum()); dl = int((fc > 0).sum())
    pct, plo, phi, pb = binom(dw, dl)
    tt = stats.ttest_ind(fc, fcc, equal_var=False).pvalue
    print("    |line| %4.1f-%4.1f  W1 n=%3d fav cover %+.2f (se %.2f) dog ATS %d-%d %.3f [%.2f,%.2f] | W5+ n=%4d fav cover %+.2f (se %.2f) | W1 vs W5+ p=%.2f" %
          (lo_, hi_, n, mu, se, dw, dl, pct, plo, phi, len(c), fcc.mean(), fcc.std() / np.sqrt(len(c)), tt))

print("\n  Week-1 home bias by home/away-favourite status (1999-2025)")
for lab, mask in [("home favoured", w1.fav_sgn > 0), ("away favoured", w1.fav_sgn < 0)]:
    x = w1[mask]; n, mu, se, p = desc(x.err_mkt); hw = int((x.err_mkt > 0).sum()); hl = int((x.err_mkt < 0).sum())
    print("    %-14s n=%3d home resid %+.2f (se %.2f) p=%.2f | home ATS %d-%d %.3f" % (lab, n, mu, se, p, hw, hl, hw / (hw + hl)))

# Levene / variance: is W1 outcome variance vs the line larger?
print("\n  Residual SD vs market by bucket (confidence-tag input) and Levene p vs W5+:")
c5 = m[m.week >= 5].err_mkt
for w in BUCKETS:
    x = m[m.wk == w].err_mkt
    print("    wk %-4s n=%4d SD err_mkt %.2f | MAE %.2f | Levene p vs W5+ = %.3f | tot_err SD %.2f (W5+ %.2f)" %
          (w, len(x), x.std(), x.abs().mean(), stats.levene(x, c5).pvalue, m[m.wk == w].tot_err.std(), m[m.week >= 5].tot_err.std()))

print("\nD. Week-by-week (2009-2025 with nfelo): n, MAE mkt / nraw / elo_line, slope of margin on elo_line, |line| mkt, |line| elo_line")
for w, x in d.groupby("week"):
    b, se, r2 = slope(x.margin, x.elo_line)
    print("  wk %2d n=%4d mkt %.2f nraw %.2f elo_line %.2f | slope elo_line %.2f (se %.2f) | mean|mkt| %.2f mean|elo_line| %.2f | SD(mkt-elo_line) %.2f" %
          (w, len(x), x.err_mkt.abs().mean(), x.err_nraw.abs().mean(), x.err_elo_line.abs().mean(), b, se, x.abs_line.mean(), x.elo_line.abs().mean(), (x.mkt - x.elo_line).std()))
