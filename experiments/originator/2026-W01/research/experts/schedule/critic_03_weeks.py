"""CRITIC 03 - T5a (week 1) and T5b (final week).
(1) engine-minus-market paired |error| difference in the final week with se; difference-of-differences vs weeks 4-16 (Welch);
    week-17 era (2009-20) vs week-18 era (2021-25).
(2) engine ATS vs close by week bucket (does the engine PICK worse in the final week, or only publish a noisier number?).
(3) leakage: the recommended rule shrinks to the CLOSE. Test the same shrink toward the market OPEN (knowable when publishing),
    and measure open->close movement in the final week vs other weeks.
(4) is the final week special? Apply the same a=0.25 shrink in EVERY week on the test set and compare the per-game gain by bucket;
    fit-set MAE-optimal a per bucket; rolling-origin choice of a for the final week.
(5) week 1: same paired test with CI; power note."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from scipy import stats
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_nfelo
_n = load_nfelo()[["gid", "home_line_open"]].drop_duplicates("gid")
from common import build, desc, ols, ats_side
pd.set_option("display.width", 250)
rng = np.random.default_rng(3)
m = build().merge(_n, on='gid', how='left'); d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["final"] = ((d.season >= 2021) & (d.week == 18)) | ((d.season <= 2020) & (d.week == 17))
d["penult"] = ((d.season >= 2021) & (d.week == 17)) | ((d.season <= 2020) & (d.week == 16))
d["wk"] = np.select([d.week == 1, d.week.between(2, 3), d.week.between(4, 16) & ~d.penult, d.penult, d.final], ["1", "2-3", "4-16", "penult", "final"], default="other")
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["proxy"] = 0.46 * (d.nraw_line + d.nfelo_bye_pts + d.nfelo_tz_pts) + 0.54 * d.rate_line - 0.15 * d.rd    # expert's candidate (c1)
d["err_proxy"] = d.margin + d.proxy
d["mkt_open"] = d.home_line_open            # nfelo file: negative = home favored (README) -> same sign as mkt_spread; verified in critic_03b
d["err_open"] = d.margin + d.mkt_open
d["dd_nraw"] = d.err_nraw.abs() - d.err_mkt.abs(); d["dd_proxy"] = d.err_proxy.abs() - d.err_mkt.abs()

print("=== (1) engine minus market |error| (paired), by bucket ===")
for lab, x in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test]), ("2009-20 (wk17 final)", d[d.season <= 2020]), ("2021-25 (wk18 final)", d[d.season >= 2021])]:
    cells = []
    for w in ["1", "4-16", "penult", "final"]:
        y = x[x.wk == w]; n, mu, se, p = desc(y.dd_nraw); cells.append("%s n=%4d %+.3f (se %.3f)" % (w, n, mu, se))
    a, b = x[x.wk == "final"].dd_nraw, x[x.wk == "4-16"].dd_nraw
    print("  %-22s " % lab + " | ".join(cells) + " | final vs 4-16 diff %+.3f Welch p=%.3f" % (a.mean() - b.mean(), stats.ttest_ind(a, b, equal_var=False).pvalue))
print("  same with the ORIGINATOR proxy (c1) instead of nfelo raw:")
for lab, x in [("ALL", d), ("TEST", d[d.test])]:
    a, b = x[x.wk == "final"].dd_proxy, x[x.wk == "4-16"].dd_proxy
    print("  %-22s final %+.3f (se %.3f, n=%d) | 4-16 %+.3f (se %.3f) | diff %+.3f Welch p=%.3f" % (lab, a.mean(), a.std() / np.sqrt(len(a)), len(a), b.mean(), b.std() / np.sqrt(len(b)), a.mean() - b.mean(), stats.ttest_ind(a, b, equal_var=False).pvalue))

print("\n=== (2) engine (nfelo raw) ATS vs close by bucket: does the engine pick worse in the final week? ===")
for lab, x in [("ALL", d), ("TEST", d[d.test])]:
    cells = []
    for w in ["1", "2-3", "4-16", "penult", "final"]:
        y = x[x.wk == w]; side = np.sign(y.mkt_spread - y.nraw_line)      # +1: engine likes home more than market
        w_, l_, p_, pct, pv = ats_side(y.err_mkt[side != 0], side[side != 0]); cells.append("%s %d-%d %.3f (p=%.2f)" % (w, w_, l_, pct, pv))
    print("  %-5s " % lab + " | ".join(cells))
    # picks with |engine - market| >= 2
    y = x[x.final & ((x.mkt_spread - x.nraw_line).abs() >= 2)]; side = np.sign(y.mkt_spread - y.nraw_line); w_, l_, p_, pct, pv = ats_side(y.err_mkt, side)
    y2 = x[(x.wk == "4-16") & ((x.mkt_spread - x.nraw_line).abs() >= 2)]; side2 = np.sign(y2.mkt_spread - y2.nraw_line); w2, l2, p2, pct2, pv2 = ats_side(y2.err_mkt, side2)
    print("        |engine-market|>=2: final %d-%d %.3f (p=%.2f) vs wk4-16 %d-%d %.3f (p=%.2f)" % (w_, l_, pct, pv, w2, l2, pct2, pv2))

print("\n=== (3) leakage check: shrink toward the market OPEN instead of the CLOSE ===")
x = d.dropna(subset=["mkt_open"])
print("  open available n=%d | sign check corr(mkt_open, margin) = %.3f (should be strongly negative) | corr(open, close) = %.3f" % (len(x), np.corrcoef(x.mkt_open, x.margin)[0, 1], np.corrcoef(x.mkt_open, x.mkt_spread)[0, 1]))
for lab, y in [("ALL", x), ("TEST", x[x.test])]:
    for w in ["4-16", "final"]:
        z = y[y.wk == w]; mv = (z.mkt_spread - z.mkt_open).abs()
        out = "  %-4s %-6s n=%4d MAE: open %.3f close %.3f proxy %.3f | |close-open| mean %.2f (>=1: %.2f)" % (lab, w, len(z), z.err_open.abs().mean(), z.err_mkt.abs().mean(), z.err_proxy.abs().mean(), mv.mean(), (mv >= 1).mean())
        for a in [0.5, 0.25]:
            e_c = z.margin + a * z.proxy + (1 - a) * z.mkt_spread; e_o = z.margin + a * z.proxy + (1 - a) * z.mkt_open
            out += " | a=%.2f: ->close %.3f ->open %.3f" % (a, e_c.abs().mean(), e_o.abs().mean())
        print(out)

print("\n=== (4) is the final week special? a=0.25 shrink (toward close) applied in EVERY bucket, test 2022-25: per-game MAE gain by bucket ===")
t = d[d.test]
for w in ["1", "2-3", "4-16", "penult", "final", "ALL"]:
    z = t if w == "ALL" else t[t.wk == w]; e0 = z.err_proxy.abs(); e1 = (z.margin + 0.25 * z.proxy + 0.75 * z.mkt_spread).abs(); dif = e1 - e0
    print("  %-6s n=%4d proxy MAE %.3f -> shrunk %.3f (gain %+.3f, se %.3f) | contribution to full-test MAE %+.4f" % (w, len(z), e0.mean(), e1.mean(), dif.mean(), dif.std() / np.sqrt(len(dif)), dif.sum() / len(t)))
print("  fit-set (<=2021) MAE-optimal engine weight a by bucket (grid 0..1):")
f = d[d.fit]
for w in ["1", "2-3", "4-16", "penult", "final"]:
    z = f[f.wk == w]; grid = np.arange(0, 1.01, 0.05); maes = [(z.margin + a * z.proxy + (1 - a) * z.mkt_spread).abs().mean() for a in grid]
    print("    %-6s n=%4d best a=%.2f (MAE %.3f vs a=1 %.3f vs a=0 %.3f)" % (w, len(z), grid[int(np.argmin(maes))], min(maes), maes[-1], maes[0]))
print("  rolling-origin for the final week: a fit on prior seasons' final weeks, applied to season s")
rows = []
for s in range(2014, 2026):
    zf = d[(d.season < s) & d.final]; zt = d[(d.season == s) & d.final]; grid = np.arange(0, 1.01, 0.05)
    a = grid[int(np.argmin([(zf.margin + g_ * zf.proxy + (1 - g_) * zf.mkt_spread).abs().mean() for g_ in grid]))]
    rows.append((s, a, len(zt), zt.err_proxy.abs().mean(), (zt.margin + a * zt.proxy + (1 - a) * zt.mkt_spread).abs().mean(), zt.err_mkt.abs().mean()))
R = pd.DataFrame(rows, columns=["season", "a_fit", "n", "MAE_proxy", "MAE_shrunk", "MAE_mkt"]); print("    " + R.round(3).to_string(index=False).replace("\n", "\n    "))

print("\n=== (5) week 1 (T5a): paired engine-minus-market with CI; also proxy-based ===")
for lab, x in [("ALL", d), ("TEST", d[d.test])]:
    a = x[x.wk == "1"]; b = x[x.wk == "4-16"]
    for c in ["dd_nraw", "dd_proxy"]:
        n, mu, se, p = desc(a[c]); n2, mu2, se2, p2 = desc(b[c])
        print("  %-4s %-8s wk1 n=%3d %+.3f 95%% CI [%+.3f, %+.3f] | wk4-16 %+.3f (se %.3f) | diff %+.3f Welch p=%.2f" % (lab, c, n, mu, mu - 1.96 * se, mu + 1.96 * se, mu2, se2, mu - mu2, stats.ttest_ind(a[c], b[c], equal_var=False).pvalue))
print("  week-1 residual SD vs market (all): %.2f vs wk4-16 %.2f; vs proxy: %.2f vs %.2f" % (d[d.wk == "1"].err_mkt.std(), d[d.wk == "4-16"].err_mkt.std(), d[d.wk == "1"].err_proxy.std(), d[d.wk == "4-16"].err_proxy.std()))
print("  NOTE: this tests nfelo's week-1 ratings (preseason-regressed with QB adjustments). PFF/Cole week-1 projections have no local history and are untested.")
