"""THEORY 4 (cont) - per-season detail of the 3-zone traveller effect, market/nfelo pricing by era, and OOS 2022-25
evaluation (paired bootstrap) of candidate travel rules on the ORIGINATOR proxy: (i) +1.0 to the 3-zone traveller,
(ii) +1.0 to the more-western team in primetime, (iii) both."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["xc"] = ((d.tz_diff.abs() == 3) & ~d.neutral)
d["west_sgn"] = np.where(d.primetime & (d.tz_diff != 0) & ~d.neutral, -np.sign(d.tz_diff), 0.0)   # +1 = home is the more-western team
co, r = ols(d.mkt_spread, [d.rate_line], ["rate_line"]); d["mkt_move"] = -(d.mkt_spread - (r.params[0] + r.params[1] * d.rate_line))  # + = market leans home vs ratings
d["lg_rate"] = d.groupby("season").err_rate.transform("mean"); d["excess_rate"] = d.err_rate - d.lg_rate

print("A. 3-zone traveller games by season: home-perspective residuals; 'excess' = err_rate minus that season's league mean err_rate")
for s, x in d[d.xc].groupby("season"):
    w, l, p_, pct, pv = ats_side(x.err_mkt, -1)
    print("  %d n=%2d err_mkt %+5.2f err_rate %+5.2f excess %+5.2f | market lean to home %+.2f nfelo tz %+.2f | traveller ATS %2d-%2d" % (s, len(x), x.err_mkt.mean(), x.err_rate.mean(), x.excess_rate.mean(), x.mkt_move.mean(), x.nfelo_tz_pts.mean(), w, l))
print("  by era: n, err_mkt, excess_rate (se), market lean, nfelo tz, traveller ATS")
for lo, hi in [(2009, 2013), (2014, 2017), (2018, 2021), (2022, 2025), (2018, 2025)]:
    x = d[d.xc & d.season.between(lo, hi)]; n, mu, se, p = desc(x.excess_rate); w, l, p_, pct, pv = ats_side(x.err_mkt, -1)
    print("  %d-%d n=%3d err_mkt %+.2f (se %.2f) excess_rate %+.2f (se %.2f p=%.3f) | market lean %+.2f nfelo tz %+.2f | traveller ATS %d-%d %.3f (p=%.3f)" % (lo, hi, len(x), x.err_mkt.mean(), desc(x.err_mkt)[2], mu, se, p, x.mkt_move.mean(), x.nfelo_tz_pts.mean(), w, l, pct, pv))
print("  2020 (no fans) excluded, 2018-2025: n=%d err_mkt %+.2f (se %.2f)" % (len(d[d.xc & d.season.between(2018, 2025) & (d.season != 2020)]), *desc(d[d.xc & d.season.between(2018, 2025) & (d.season != 2020)].err_mkt)[1:3]))
# 2-zone games for contrast
x = d[(d.tz_diff.abs() == 2) & ~d.neutral]; print("  contrast 2-zone games all seasons n=%d err_mkt %+.2f (se %.2f); 2018-25 n=%d err_mkt %+.2f (se %.2f)" % (len(x), *desc(x.err_mkt)[1:3], len(x[x.season >= 2018]), *desc(x[x.season >= 2018].err_mkt)[1:3]))

print("\nB. primetime west edge by season-era (more-western team vs market), and overlap with 3-zone rule")
for lo, hi in [(2009, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]:
    x = d[(d.west_sgn != 0) & d.season.between(lo, hi)]; n, mu, se, p = desc(x.err_mkt * x.west_sgn); w, l, p_, pct, pv = ats_side(x.err_mkt, x.west_sgn)
    print("  %d-%d n=%3d west edge vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f | west ATS %d-%d %.3f" % (lo, hi, n, mu, se, p, (x.err_rate * x.west_sgn).mean(), w, l, pct))
print("  games with both flags (primetime & 3 zones): %d of %d xc / %d primetime-tz games" % ((d.xc & (d.west_sgn != 0)).sum(), d.xc.sum(), (d.west_sgn != 0).sum()))

print("\nC. OOS 2022-25 on the ORIGINATOR proxy (0.46 nfelo raw stripped of bye/tz mods + 0.54 rating-only, + rest 0.15/day) = base (c1)")
t = d[d.test].copy(); rng = np.random.default_rng(1)
base = 0.46 * (t.nraw_line + t.nfelo_bye_pts + t.nfelo_tz_pts) + 0.54 * t.rate_line - 0.15 * t.rd
def ev(name, line, aff):
    e = t.margin + line; e0 = t.margin + base; diff = e.abs().values - e0.abs().values
    bs = np.array([rng.choice(diff, len(diff)).mean() for _ in range(4000)])
    w, l, p_, pct, pv = ats_side(t.err_mkt[aff], np.sign(t.mkt_spread[aff] - line[aff])) if aff.sum() else (0, 0, 0, np.nan, np.nan)
    print("  %-52s MAE %.4f (d=%+.4f, 95%% CI [%+.4f, %+.4f]) | affected n=%3d MAE %.3f bias %+.2f | ATS vs close %d-%d %.3f" %
          (name, e.abs().mean(), diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), aff.sum(), e[aff].abs().mean(), e[aff].mean(), w, l, pct))
    return e
ev("(c1) base", base, t.xc & False)
for k in [0.5, 1.0, 1.5, 2.0]:
    ev("+%.1f to 3-zone traveller" % k, base + k * t.xc, t.xc)
for k in [0.5, 1.0, 1.5]:
    ev("+%.1f to more-western team in primetime" % k, base - k * t.west_sgn, t.west_sgn != 0)
ev("both: +1.0 traveller & +1.0 west primetime", base + 1.0 * t.xc - 1.0 * t.west_sgn, t.xc | (t.west_sgn != 0))
we = ((t.away_off <= -2) & t.early & ~t.neutral).astype(float)
ev("spec as written: -0.6 to MT/PT away team at 1pm ET", base - 0.6 * we, we == 1)   # ORIGINATOR sign: line more negative = home favored more
ev("reverse of spec: +0.6 to MT/PT away team at 1pm ET", base + 0.6 * we, we == 1)
ev("nfelo's own tz mod re-applied to whole blend", base - t.nfelo_tz_pts, t.nfelo_tz_pts != 0)
print("  per-season MAE change of '+1.0 to 3-zone traveller' (affected games):")
for s, x in t.groupby("season"):
    xx = x[x.xc]; e0 = (xx.margin + base[xx.index]).abs().mean(); e1 = (xx.margin + base[xx.index] + 1.0).abs().mean()
    print("    %d n=%2d MAE %.3f -> %.3f (d=%+.3f) bias(home) %+.2f" % (s, len(xx), e0, e1, e1 - e0, (xx.margin + base[xx.index]).mean()))
# fit-window sensitivity for the traveller k: fit on 2009-21 vs 2018-21
for lo in [2009, 2014, 2018]:
    f = d[d.xc & d.season.between(lo, 2021)]; ks = np.arange(-1, 4.01, 0.25)
    k = ks[int(np.argmin([(f.margin + f.rate_line + kk).abs().mean() for kk in ks]))]
    print("  fit window %d-2021 (n=%d): MAE-min k for traveller = %+.2f; mean err_rate %+.2f" % (lo, len(f), k, f.err_rate.mean()))
