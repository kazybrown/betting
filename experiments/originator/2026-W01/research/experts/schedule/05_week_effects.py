"""THEORY 5 - Week-of-season effects: Week 1 (rating uncertainty) and the final week (17 pre-2021, 18 from 2021: rest/motivation).
Measures MAE/bias of market vs nfelo raw/close/rating-only by week bucket, favorite behaviour, totals. Fit <=2021, test 2022-25."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, desc, ols, ats_side
from scipy import stats
pd.set_option("display.width", 250)
m = build()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["final"] = ((d.season >= 2021) & (d.week == 18)) | ((d.season <= 2020) & (d.week == 17))
d["penult"] = ((d.season >= 2021) & (d.week == 17)) | ((d.season <= 2020) & (d.week == 16))
d["wk"] = np.select([d.week == 1, d.week == 2, d.week == 3, d.week.between(4, 8), d.penult, d.final], ["1", "2", "3", "4-8", "penult", "final"], default="9-16")
order = ["1", "2", "3", "4-8", "9-16", "penult", "final"]
d["fav_sgn"] = np.sign(-d.mkt_spread)                   # +1 home favored
d["fav_cover"] = d.err_mkt * d.fav_sgn                  # + = favorite covered by this much
d["abs_line"] = d.mkt_spread.abs()

def block(title, dd):
    print("\n" + title)
    print("  %-7s %5s | MAE: market nraw  nclose rate | nraw-mkt | SD err_mkt | bias home %-14s | fav cover %-14s | dog ATS | |line| mkt / rate | tot resid" % ("wk", "n", "", ""))
    for w in order:
        x = dd[dd.wk == w]
        if len(x) < 5: continue
        n, mu, se, p = desc(x.err_mkt); n2, mu2, se2, p2 = desc(x.fav_cover[x.fav_sgn != 0])
        wn, ln, ps, pct, pv = ats_side(x.err_mkt[x.fav_sgn != 0], -x.fav_sgn[x.fav_sgn != 0])
        tn, tmu, tse, tp = desc(x.tot_err_mkt)
        print("  %-7s %5d | %.2f  %.2f  %.2f  %.2f | %+.2f | %.2f | %+.2f (se %.2f) | %+.2f (se %.2f) | %d-%d %.3f | %.2f / %.2f | %+.2f (se %.2f)" %
              (w, len(x), x.err_mkt.abs().mean(), x.err_nraw.abs().mean(), x.err_nclose.abs().mean(), x.err_rate.abs().mean(),
               x.err_nraw.abs().mean() - x.err_mkt.abs().mean(), x.err_mkt.std(), mu, se, mu2, se2, wn, ln, pct, x.abs_line.mean(), x.rate_line.abs().mean(), tmu, tse))
block("A. ALL 2009-2025 by week bucket", d)
block("A. FIT 2009-2021", d[d.fit])
block("A. TEST 2022-2025", d[d.test])

# --- B. Week 1: is the engine's disadvantage vs the market larger than in mid-season? (paired diff of abs errors)
print("\nB. Week 1 vs weeks 4-16: engine (nfelo raw) minus market abs error; and how much shrinking the engine toward the market helps")
for lab, dd in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test])]:
    for w, mask in [("wk1", dd.week == 1), ("wk2-3", dd.week.between(2, 3)), ("wk4-16", dd.week.between(4, 16))]:
        x = dd[mask]; dif = x.err_nraw.abs() - x.err_mkt.abs(); n, mu, se, p = desc(dif)
        # shrink: line = a*nraw + (1-a)*mkt ; best a on this bucket (descriptive), plus MAE at a=0.5
        best = min(np.arange(0, 1.01, 0.05), key=lambda a: (x.margin + a * x.nraw_line + (1 - a) * x.mkt_spread).abs().mean())
        print("  %-4s %-6s n=%4d  MAE nraw-mkt %+.3f (se %.3f p=%.3f) | corr(nraw,mkt) %.3f | SD(nraw-mkt) %.2f | best engine weight a=%.2f" %
              (lab, w, n, mu, se, p, np.corrcoef(x.nraw_line, x.mkt_spread)[0, 1], (x.nraw_line - x.mkt_spread).std(), best))
# variance of residual for confidence tags: does |err| in week 1 differ?  Levene test wk1 vs wk4-16 on err_mkt and err_nraw
for c in ["err_mkt", "err_nraw"]:
    a = d[d.week == 1][c]; b = d[d.week.between(4, 16)][c]
    print("  Levene %s wk1 (sd %.2f, n=%d) vs wk4-16 (sd %.2f, n=%d): p=%.3f" % (c, a.std(), len(a), b.std(), len(b), stats.levene(a, b).pvalue))

# --- C. Final week: favorites by size, totals, market-vs-rating gap (news the engine cannot see)
print("\nC. Final week (17 pre-2021 / 18 from 2021): favourite cover margin by line size; totals; |market - rating line| gap")
d["gap"] = (d.mkt_spread - d.rate_line).abs()
for lab, dd in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test])]:
    x = dd[dd.final]; o = dd[dd.week.between(4, 16)]
    print("  %-4s final n=%3d: fav cover %+.2f (se %.2f) | dog ATS %s | tot resid %+.2f (se %.2f) | MAE mkt %.2f nraw %.2f | gap |mkt-rate| %.2f vs wk4-16 %.2f (p=%.3f) | home bias %+.2f" %
          (lab, len(x), desc(x.fav_cover[x.fav_sgn != 0])[1], desc(x.fav_cover[x.fav_sgn != 0])[2], "%d-%d" % ats_side(x.err_mkt[x.fav_sgn != 0], -x.fav_sgn[x.fav_sgn != 0])[:2],
           desc(x.tot_err_mkt)[1], desc(x.tot_err_mkt)[2], x.err_mkt.abs().mean(), x.err_nraw.abs().mean(), x.gap.mean(), o.gap.mean(), stats.ttest_ind(x.gap, o.gap, equal_var=False).pvalue, x.err_mkt.mean()))
    for lo, hi in [(0, 3), (3.5, 6.5), (7, 30)]:
        y = x[x.abs_line.between(lo, hi) & (x.fav_sgn != 0)]
        n, mu, se, p = desc(y.fav_cover); n2, mu2, se2, p2 = desc(y.err_nraw * y.fav_sgn)
        print("       |line| %4.1f-%4.1f n=%3d fav cover vs mkt %+.2f (se %.2f p=%.2f) | fav vs nfelo raw %+.2f (se %.2f) | mkt fav ATS %d-%d" % (lo, hi, n, mu, se, p, mu2, se2, *ats_side(y.err_mkt, y.fav_sgn)[:2]))
# Does the gap predict? in final week, when the market deviates from the ratings, who is right? regress margin on rate_line & mkt_spread (final vs mid)
print("  who is right when market and ratings disagree (OLS margin ~ -rate_line, -mkt_spread; weights):")
for lab, mask in [("final ALL", d.final), ("final TEST", d.final & d.test), ("wk4-16 ALL", d.week.between(4, 16)), ("wk1 ALL", d.week == 1), ("wk1 TEST", (d.week == 1) & d.test)]:
    x = d[mask]; co, r = ols(x.margin, [-x.rate_line, -x.mkt_spread], ["rating", "market"])
    print("   %-11s n=%4d rating w=%+.2f (se %.2f) market w=%+.2f (se %.2f)" % (lab, len(x), co["rating"][0], co["rating"][1], co["market"][0], co["market"][1]))

# --- D. week-by-week MAE (market, nraw) table for the appendix
print("\nD. week-by-week (ALL): n, MAE market, MAE nraw, diff, SD err_mkt, bias home, |line| market")
for w, x in d.groupby("week"):
    print("  wk %2d n=%4d mkt %.2f nraw %.2f diff %+.2f sd %.2f home bias %+.2f |line| %.2f" % (w, len(x), x.err_mkt.abs().mean(), x.err_nraw.abs().mean(), x.err_nraw.abs().mean() - x.err_mkt.abs().mean(), x.err_mkt.std(), x.err_mkt.mean(), x.abs_line.mean()))
