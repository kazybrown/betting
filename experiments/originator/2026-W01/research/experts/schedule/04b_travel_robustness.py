"""THEORY 4 (cont) - robustness of the cross-country result (away team crossing 3 zones beats the line):
era splits, team fixed effects, leave-one-team-out, west-team quality confound, primetime circadian term, wider k grid."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import build, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); m = m[~m.neutral].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["xc"] = (d.tz_diff.abs() == 3).astype(float); d["xc_east"] = (d.tz_diff == 3).astype(float); d["xc_west"] = (d.tz_diff == -3).astype(float)
d["two"] = (d.tz_diff.abs() == 2).astype(float); d["dist_k"] = d.dist / 1000
d["west_edge_mkt"] = -d.err_mkt * np.sign(d.tz_diff); d["west_edge_rate"] = -d.err_rate * np.sign(d.tz_diff)   # + = the more-western team beat the line
print("xc games n=%d (fit %d, test %d)" % (d.xc.sum(), (d.xc * d.fit).sum(), (d.xc * d.test).sum()))

print("\nA. cross-country (|tz_diff|=3) by era: home-perspective residuals (+ = home beat line); ATS = bet the traveller")
for lo, hi in [(2009, 2012), (2013, 2017), (2018, 2021), (2022, 2025)]:
    x = d[(d.xc == 1) & d.season.between(lo, hi)]
    w, l, p_, pct, pv = ats_side(x.err_mkt, -1)
    print("  %d-%d n=%3d err_mkt %+.2f (se %.2f) err_rate %+.2f (se %.2f) err_nraw %+.2f | traveller ATS %d-%d %.3f" % (lo, hi, len(x), desc(x.err_mkt)[1], desc(x.err_mkt)[2], desc(x.err_rate)[1], desc(x.err_rate)[2], desc(x.err_nraw)[1], w, l, pct))

print("\nB. team fixed effects: err ~ xc (+ home & away team FE, season FE). HC1 se")
for y in ["err_mkt", "err_rate", "err_nraw"]:
    for form, lab in [("%s ~ xc" % y, "no FE"), ("%s ~ xc + C(home_team) + C(away_team)" % y, "team FE"),
                      ("%s ~ xc + C(home_team) + C(away_team) + C(season)" % y, "team+season FE"),
                      ("%s ~ xc_east + xc_west + two + C(home_team) + C(away_team)" % y, "east/west split, team FE"),
                      ("%s ~ dist_k + C(home_team) + C(away_team)" % y, "distance, team FE")]:
        r = smf.ols(form, data=d).fit(cov_type="HC1")
        keys = [k for k in r.params.index if k in ("xc", "xc_east", "xc_west", "two", "dist_k")]
        print("  %-8s %-26s " % (y, lab) + "  ".join("%s=%+.2f (se %.2f, p=%.2f)" % (k, r.params[k], r.bse[k], r.pvalues[k]) for k in keys))
# fit/test with team FE
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test])]:
    r = smf.ols("err_mkt ~ xc + C(home_team) + C(away_team)", data=dd).fit(cov_type="HC1")
    r2 = smf.ols("err_rate ~ xc + C(home_team) + C(away_team)", data=dd).fit(cov_type="HC1")
    print("  %-4s team FE: err_mkt xc=%+.2f (se %.2f p=%.2f) | err_rate xc=%+.2f (se %.2f p=%.2f)" % (lab, r.params["xc"], r.bse["xc"], r.pvalues["xc"], r2.params["xc"], r2.bse["xc"], r2.pvalues["xc"]))

print("\nC. leave-one-team-out (drop all xc games involving team T): mean err_mkt of remaining xc games")
xc = d[d.xc == 1]
teams = sorted(set(xc.home_team) | set(xc.away_team)); res = []
for t in teams:
    x = xc[(xc.home_team != t) & (xc.away_team != t)]; res.append((t, len(x), x.err_mkt.mean(), x.err_rate.mean()))
res = pd.DataFrame(res, columns=["drop", "n", "err_mkt", "err_rate"])
print("  range of err_mkt over LOTO: %.2f .. %.2f (base %.2f); err_rate %.2f .. %.2f" % (res.err_mkt.min(), res.err_mkt.max(), xc.err_mkt.mean(), res.err_rate.min(), res.err_rate.max()))
print("  most influential drops:", res.reindex(res.err_mkt.abs().argsort()).head(4).round(2).to_dict("records"))
print("  by away (traveller) team, n>=12:")
for t, x in xc.groupby("away_team"):
    if len(x) >= 12: print("    away %-4s n=%3d err_mkt %+.2f err_rate %+.2f | traveller ATS %d-%d" % (t, len(x), x.err_mkt.mean(), x.err_rate.mean(), *ats_side(x.err_mkt, -1)[:2]))
print("  by home team, n>=12:")
for t, x in xc.groupby("home_team"):
    if len(x) >= 12: print("    home %-4s n=%3d err_mkt %+.2f err_rate %+.2f" % (t, len(x), x.err_mkt.mean(), x.err_rate.mean()))

print("\nD. west-team quality confound: PT/MT franchises' residual in xc games vs their other games (perspective of the west team)")
d["west_home"] = d.home_off <= -2; d["west_away"] = d.away_off <= -2
wa = d[d.west_away]; wh = d[d.west_home]
for lab, x, sgn in [("west team AWAY, xc (3 zones)", wa[wa.tz_diff == 3], -1), ("west team AWAY, 1-2 zones", wa[wa.tz_diff.between(1, 2)], -1), ("west team AWAY, same zone", wa[wa.tz_diff == 0], -1),
                    ("west team HOME, xc visitor", wh[wh.tz_diff == -3], 1), ("west team HOME, 1-2 zone visitor", wh[wh.tz_diff.between(-2, -1)], 1), ("west team HOME, same zone", wh[wh.tz_diff == 0], 1)]:
    n, mu, se, p = desc(x.err_mkt * sgn); n2, mu2, se2, p2 = desc(x.err_rate * sgn)
    print("  %-34s n=%4d west team vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f (se %.2f)" % (lab, n, mu, se, p, mu2, se2))
# east team playing out west (ET franchise away at MT/PT host) vs ET franchise away elsewhere
ea = d[d.away_off == 0]
for lab, x in [("ET team AWAY at PT/MT host", ea[ea.home_off <= -2]), ("ET team AWAY at CT host", ea[ea.home_off == -1]), ("ET team AWAY at ET host", ea[ea.home_off == 0])]:
    n, mu, se, p = desc(-x.err_mkt); print("  %-34s n=%4d east team vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f" % (lab, n, mu, se, p, -x.err_rate.mean()))

print("\nE. primetime circadian: the more-western team's edge vs the line, primetime (kick>=19:00 ET) vs day games, by zones apart")
for lab, mask in [("primetime", d.primetime), ("day games (placebo)", ~d.primetime)]:
    for z in [1, 2, 3]:
        x = d[mask & (d.tz_diff.abs() == z)]
        n, mu, se, p = desc(x.west_edge_mkt); n2, mu2, se2, p2 = desc(x.west_edge_rate)
        w, l, p_, pct, pv = ats_side(x.err_mkt, -np.sign(x.tz_diff))
        print("  %-20s %d zone(s) n=%4d west edge vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f (se %.2f) | west ATS %d-%d %.3f | nfelo tz mod (west persp.) %+.2f" %
              (lab, z, n, mu, se, p, mu2, se2, w, l, pct, (-x.nfelo_tz_pts * np.sign(x.tz_diff)).mean()))
    x = d[mask & (d.tz_diff != 0)]; co, r = ols(x.west_edge_mkt, [x.tz_diff.abs().astype(float)], ["zones"]); co2, r2 = ols(x.west_edge_rate, [x.tz_diff.abs().astype(float)], ["zones"])
    print("  %-20s slope per zone: vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f (se %.2f p=%.2f), n=%d" % (lab, *co["zones"], *co2["zones"], len(x)))
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test])]:
    x = dd[dd.primetime & (dd.tz_diff != 0)]; n, mu, se, p = desc(x.west_edge_mkt); w, l, p_, pct, pv = ats_side(x.err_mkt, -np.sign(x.tz_diff))
    print("  %-4s primetime any zones apart: n=%3d west edge vs market %+.2f (se %.2f p=%.2f) | west ATS %d-%d %.3f (p=%.2f)" % (lab, n, mu, se, p, w, l, pct, pv))

print("\nF. OOS: adjustment k (pts to the TRAVELLER) on xc games, fit on 2009-21 with a wide grid; test 2022-25")
f = d[(d.fit) & (d.xc == 1)]; t = d[(d.test) & (d.xc == 1)]
ks = np.arange(-1, 4.01, 0.25); fm = [(f.margin + f.rate_line + k).abs().mean() for k in ks]   # +k to the line = toward the away team
kb = ks[int(np.argmin(fm))]; print("  fit-set n=%d MAE-min k=%.2f (MAE %.3f vs k=0 %.3f); fit mean err_rate %+.2f" % (len(f), kb, min(fm), fm[0], f.err_rate.mean()))
for k in [0, 0.5, 1.0, 1.5, 2.0, kb]:
    e = t.margin + t.rate_line + k; print("  test xc n=%d k=%.2f MAE %.3f RMSE %.3f bias(home) %+.2f" % (len(t), k, e.abs().mean(), np.sqrt((e**2).mean()), e.mean()))
print("  test market MAE on xc games %.3f, nfelo raw %.3f" % (t.err_mkt.abs().mean(), t.err_nraw.abs().mean()))
