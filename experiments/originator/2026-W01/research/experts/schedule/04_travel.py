"""THEORY 4 - Travel / time zones: west-coast teams at 1pm ET (body clock 10am), east teams in late/prime windows
out west, cross-country trips, distance. Team time zones + stadium coords defined in common.py (gametime is ET).
nfelo already applies home_time_advantage_mod. Non-neutral REG games; fit <=2021, test 2022-25."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, report_means, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); m = m[~m.neutral].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
print("sample non-neutral REG with nfelo: n=%d fit %d test %d" % (len(d), d.fit.sum(), d.test.sum()))
# features (away team perspective unless noted)
d["away_pt_early"] = (d.away_off == -3) & d.early                       # PT team kicking at 1pm ET (10am body)
d["away_mt_early"] = (d.away_off == -2) & d.early                       # MT team at 1pm ET (11am body)
d["away_west_early"] = (d.away_off <= -2) & d.early
d["away_east_late_west"] = (d.away_off == 0) & (d.home_off <= -2) & d.late   # east team, 4pm ET window, hosted out west
d["away_east_prime_west"] = (d.away_off == 0) & (d.home_off <= -2) & d.primetime
d["away_west_prime_east"] = (d.away_off <= -2) & (d.home_off == 0) & d.primetime
d["home_west_prime_east_visitor"] = d.away_east_prime_west
d["xc_east"] = d.tz_diff == 3     # away travels 3 zones east (PT team at an ET home)
d["xc_west"] = d.tz_diff == -3    # away travels 3 zones west
d["dist_k"] = d.dist / 1000.0
print("feature counts:", {c: int(d[c].sum()) for c in ["away_pt_early", "away_mt_early", "away_west_early", "away_east_late_west", "away_east_prime_west", "away_west_prime_east", "xc_east", "xc_west"]})
print("nfelo tz mod (pts, + favors home) mean by feature:", {c: round(d[d[c]].nfelo_tz_pts.mean(), 2) for c in ["away_pt_early", "away_mt_early", "away_east_late_west", "away_east_prime_west", "away_west_prime_east", "xc_east", "xc_west"]})

for lab, dd in [("ALL 2009-25", d), ("FIT 2009-21", d[d.fit]), ("TEST 2022-25", d[d.test])]:
    report_means("A. mean residual (home perspective; + = home beat the line) (%s)" % lab, dd, [
        ("away PT team @ 1pm ET (body 10am)", dd.away_pt_early),
        ("away MT team @ 1pm ET (body 11am)", dd.away_mt_early),
        ("away ET team, 4pm window, host MT/PT", dd.away_east_late_west),
        ("away ET team, primetime, host MT/PT", dd.away_east_prime_west),
        ("away MT/PT team, primetime, host ET", dd.away_west_prime_east),
        ("away crosses 3 zones east (any kick)", dd.xc_east),
        ("away crosses 3 zones west (any kick)", dd.xc_west),
        ("same time zone", dd.tz_diff == 0)])

# --- B. ATS vs close of the theory's side in each bucket (test + all)
print("\nB. ATS vs close for the side the theory favors (home side when away is disadvantaged)")
for lab, mask, side in [("fade PT team @1pm ET (bet home)", d.away_pt_early, 1), ("fade MT/PT team @1pm ET (bet home)", d.away_west_early, 1),
                        ("fade ET team late window out west (bet home)", d.away_east_late_west, 1), ("fade ET team primetime out west (bet home)", d.away_east_prime_west, 1),
                        ("back west team primetime in east (bet away)", d.away_west_prime_east, -1), ("fade 3-zone-east traveller (bet home)", d.xc_east, 1),
                        ("fade 3-zone-west traveller (bet home)", d.xc_west, 1)]:
    for sub, dd in [("ALL", d[mask]), ("FIT", d[mask & d.fit]), ("TEST", d[mask & d.test])]:
        w, l, ps, pct, pv = ats_side(dd.err_mkt, side)
        print("  %-46s %-4s n=%3d ATS %3d-%3d-%2d %.3f (p=%.2f) | mean err_mkt*side %+.2f | err_rate*side %+.2f | err_nraw*side %+.2f" %
              (lab, sub, len(dd), w, l, ps, pct, pv, (dd.err_mkt * side).mean(), (dd.err_rate * side).mean(), (dd.err_nraw * side).mean()))

# --- C. continuous: distance and tz_diff (signed), and primetime circadian term
print("\nC. OLS residual ~ dist(k miles) + tz_diff + primetime*tz_diff  (+ coef = favors HOME)  [HC1]")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    X = [dd.dist_k, dd.tz_diff.astype(float), (dd.primetime * dd.tz_diff).astype(float)]; names = ["dist_k", "tz_diff", "prime*tz_diff"]
    for c in ["err_mkt", "err_rate", "err_nraw"]:
        co, r = ols(dd[c], X, names); print("  %-4s %-8s " % (lab, c) + "  ".join("%s=%+.3f(%.3f,p=%.2f)" % (k, *co[k]) for k in names))
    co, r = ols(dd.mkt_spread, [dd.rate_line] + X, ["rate_line"] + names)
    print("  %-4s market   " % lab + "  ".join("%s=%+.3f(%.3f)" % (k, -co[k][0], co[k][1]) for k in names) + "   [market move toward home per unit]")
print("  distance-only, fit vs test (err_rate):")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test])]:
    co, r = ols(dd.err_rate, [dd.dist_k], ["dist_k"]); co2, r2 = ols(dd.err_mkt, [dd.dist_k], ["dist_k"])
    print("   %-4s err_rate slope %+.3f/1000mi (se %.3f p=%.2f) | err_mkt slope %+.3f (se %.3f p=%.2f)" % (lab, *co["dist_k"], *co2["dist_k"]))
print("  distance buckets (ALL): mean err_rate / err_mkt")
for b, x in d.groupby(pd.cut(d.dist, [-1, 300, 800, 1500, 2200, 3000]), observed=True):
    print("   %-14s n=%4d err_rate %+.2f (se %.2f) err_mkt %+.2f (se %.2f) nfelo tz mod %+.2f" % (b, len(x), desc(x.err_rate)[1], desc(x.err_rate)[2], desc(x.err_mkt)[1], desc(x.err_mkt)[2], x.nfelo_tz_pts.mean()))

# --- D. does nfelo's tz mod help OOS? compare rating-only vs rating-only + nfelo tz mod vs + fitted dummies
print("\nD. OOS 2022-25 (non-neutral): MAE of rating-only line vs + nfelo tz mod vs + fitted west-early dummy")
f = d[d.fit]; t = d[d.test]
base = t.margin + t.rate_line
e_tz = t.margin + t.rate_line - t.nfelo_tz_pts
print("  rating-only MAE %.4f | + nfelo tz mod MAE %.4f (d=%+.4f) | games with tz mod != 0: n=%d, MAE base %.4f vs +tz %.4f" %
      (base.abs().mean(), e_tz.abs().mean(), e_tz.abs().mean() - base.abs().mean(), (t.nfelo_tz_pts != 0).sum(), base[t.nfelo_tz_pts != 0].abs().mean(), e_tz[t.nfelo_tz_pts != 0].abs().mean()))
for feat_name in ["away_pt_early", "away_west_early", "xc_east", "xc_west"]:
    ks = np.arange(-2, 2.01, 0.1); fe = f[feat_name].astype(float)
    k = ks[int(np.argmin([(f.margin + f.rate_line - kk * fe).abs().mean() for kk in ks]))]   # k>0 favors home (fade the away team)
    te = t[feat_name].astype(float); e = t.margin + t.rate_line - k * te; sel = te == 1
    print("  %-16s fit k=%+.1f (pts toward home) | test games n=%3d: MAE base %.3f -> adj %.3f (d=%+.3f); full-test MAE d=%+.4f" %
          (feat_name, k, sel.sum(), base[sel].abs().mean(), e[sel].abs().mean(), e[sel].abs().mean() - base[sel].abs().mean(), e.abs().mean() - base.abs().mean()))
