"""THEORY 1 - Short rest (Thursday after Sunday = 4 days). Spec: -0.6..-1.2 for the short-rest team.
Residuals vs market close, nfelo close, nfelo raw (unregressed), rating-only line. Fit <=2021, test 2022-25."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, report_means, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); m = m[m.rest_valid].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()      # 4-series sample
print("sample: REG weeks 2+, with nfelo:", len(d), "| fit", d.fit.sum(), "| test", d.test.sum())
normal = lambda r: r.between(6, 8)
both_short = d.home_short & d.away_short
home_only = d.home_short & normal(d.away_rest)
away_only = d.away_short & normal(d.home_rest)
print("\ncounts: both short %d | home short only %d | away short only %d | home short vs away bye/long %d | away short vs home bye/long %d" %
      (both_short.sum(), home_only.sum(), away_only.sum(), (d.home_short & (d.away_rest > 8)).sum(), (d.away_short & (d.home_rest > 8)).sum()))
print("Thursday games: n=%d, of which both rest 4: %d" % ((d.weekday == "Thursday").sum(), ((d.weekday == "Thursday") & both_short).sum()))

# --- A. mean residuals by group (home perspective; + = home beat the line)
for lab, dd in [("ALL 2009-2025", d), ("FIT 2009-2021", d[d.fit]), ("TEST 2022-2025", d[d.test])]:
    report_means("A. mean residual by short-rest configuration (%s)" % lab, dd, [
        ("both short (TNF standard)", (dd.home_short & dd.away_short)),
        ("home short, away normal 6-8", dd.home_short & normal(dd.away_rest)),
        ("away short, home normal 6-8", dd.away_short & normal(dd.home_rest)),
        ("neither short", ~dd.home_short & ~dd.away_short)])

# --- B. signed 'rested team vs short team' effect: sign = +1 when away is the short one (home advantaged)
one = d[(d.home_short ^ d.away_short)].copy()
one["sgn"] = np.where(one.away_short, 1, -1)
print("\nB. rested team's residual vs the short-rest team (one team short, any opponent rest):")
for lab, dd in [("ALL", one), ("FIT", one[one.fit]), ("TEST", one[one.test])]:
    row = []
    for c in ["err_mkt", "err_nclose", "err_nraw", "err_rate"]:
        n, mu, se, p = desc(dd[c] * dd.sgn); row.append("%s %+.2f (se %.2f p=%.2f)" % (c, mu, se, p))
    w, l, ps, pct, pv = ats_side(dd.err_mkt, dd.sgn)
    print("  %-5s n=%3d | %s | ATS rested team vs close %d-%d-%d (%.3f, p=%.2f)" % (lab, len(dd), " | ".join(row), w, l, ps, pct, pv))
one2 = one[pd.Series(np.where(one.away_short, one.home_rest, one.away_rest), index=one.index).between(6, 8)]
print("  (opponent on normal 6-8 rest only) n=%d: mkt %+.2f (se %.2f) | rate %+.2f (se %.2f)" %
      (len(one2), (one2.err_mkt*one2.sgn).mean(), (one2.err_mkt*one2.sgn).std()/np.sqrt(len(one2)),
       (one2.err_rate*one2.sgn).mean(), (one2.err_rate*one2.sgn).std()/np.sqrt(len(one2))))

# --- C. regression with all rest dummies (home minus away), so the short-rest coefficient is net of byes / long rest
def rest_dummies(dd):
    return {"short(<=5)": dd.away_short.astype(float) - dd.home_short.astype(float),
            "mini(6)": dd.away_mini.astype(float) - dd.home_mini.astype(float),
            "long(9-12)": dd.home_long.astype(float) - dd.away_long.astype(float),
            "bye(13+)": dd.home_bye.astype(float) - dd.away_bye.astype(float)}
print("\nC. OLS residual ~ rest dummies (each coded so + = the rested side gains; HC1 se). Coef, se, p:")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    X = rest_dummies(dd); names = list(X)
    for c in ["err_mkt", "err_rate", "err_nraw"]:
        co, r = ols(dd[c], [X[k] for k in names], names)
        print("  %-4s %-9s " % (lab, c) + "  ".join("%s=%+.2f(%.2f,p=%.2f)" % (k, *co[k]) for k in names))

# --- D. what the market prices: mkt_spread ~ rate_line + rest dummies (coef sign: negative = market favors home more)
print("\nD. market pricing: mkt_spread ~ rate_line + rest dummies (dummies coded + = home side rested; negative coef = market moves toward the rested side)")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    X = rest_dummies(dd); names = ["rate_line"] + list(X)
    co, r = ols(dd.mkt_spread, [dd.rate_line] + [X[k] for k in list(X)], names)
    print("  %-4s " % lab + "  ".join("%s=%+.3f(%.3f)" % (k, co[k][0], co[k][1]) for k in names))

# --- E. TNF both-short: totals and variance
print("\nE. Thursday both-short games: totals and error dispersion")
for lab, dd in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test])]:
    t = dd[(dd.home_short & dd.away_short)]; o = dd[~(dd.home_short | dd.away_short) & dd.weekday.eq("Sunday")]
    n, mu, se, p = desc(t.tot_err_mkt); n2, mu2, se2, p2 = desc(o.tot_err_mkt)
    print("  %-4s TNF n=%3d total resid %+.2f (se %.2f p=%.2f) mean line %.1f realized %.1f | Sunday n=%4d resid %+.2f (se %.2f) line %.1f realized %.1f | spread MAE TNF %.2f vs Sun %.2f; over rate TNF %.3f Sun %.3f" %
          (lab, n, mu, se, p, t.mkt_total.mean(), t.total_pts.mean(), n2, mu2, se2, o.mkt_total.mean(), o.total_pts.mean(),
           t.err_mkt.abs().mean(), o.err_mkt.abs().mean(), (t.tot_err_mkt > 0).mean(), (o.tot_err_mkt > 0).mean()))
# total residual by era (TNF package expanded in 2012, 2014 CBS/NFLN, 2022 Amazon)
print("  TNF total residual by era:")
for lo, hi in [(2009, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]:
    t = d[(d.home_short & d.away_short) & d.season.between(lo, hi)]
    n, mu, se, p = desc(t.tot_err_mkt); print("    %d-%d n=%3d total resid %+.2f (se %.2f p=%.2f) | market total line mean %.1f" % (lo, hi, n, mu, se, p, t.mkt_total.mean()))

# --- F. OOS: adjustment size k applied to the rating-only line for one-team-short games; MAE on test
print("\nF. OOS 2022-25: rating-only line + k*(rested side) on one-team-short games; MAE vs results (lower better)")
t = one[one.test]
for k in [0, 0.5, 0.9, 1.2, 1.5, 2.0]:
    pred = t.rate_line - k * t.sgn     # ORIGINATOR convention: more negative = home favored; sgn=+1 home rested
    e = t.margin + pred
    print("  k=%.1f  n=%d  MAE %.3f  bias %+.2f" % (k, len(t), e.abs().mean(), e.mean()))
print("  market MAE on same games: %.3f" % t.err_mkt.abs().mean())
# fit-set optimum for k (grid) for reference
f = one[one.fit]
ks = np.arange(0, 3.01, 0.1); maes = [ (f.margin + f.rate_line - k*f.sgn).abs().mean() for k in ks]
print("  fit-set (2009-21, n=%d) MAE-minimizing k = %.1f (MAE %.3f vs k=0 %.3f); fit-set mean gross effect = %+.2f" % (len(f), ks[int(np.argmin(maes))], min(maes), maes[0], (f.err_rate*f.sgn).mean()))
