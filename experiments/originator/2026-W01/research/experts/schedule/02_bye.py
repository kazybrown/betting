"""THEORY 2 - Bye advantage (rest >= 13) vs an opponent on normal rest. Spec: +0.5..+1.0 for the bye team.
nfelo already applies home_bye_mod (~+0.7 pt) / away_bye_mod (~+1.8 pt for the away team). Fit <=2021, test 2022-25."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, report_means, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); m = m[m.rest_valid].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
normal = lambda r: r.between(6, 8)
print("sample (REG wk2+, nfelo present): n=%d fit %d test %d | market-only sample n=%d" % (len(d), d.fit.sum(), d.test.sum(), len(m)))
print("counts: home bye %d | away bye %d | both %d | home bye vs away normal %d | away bye vs home normal %d" %
      (d.home_bye.sum(), d.away_bye.sum(), (d.home_bye & d.away_bye).sum(), (d.home_bye & normal(d.away_rest)).sum(), (d.away_bye & normal(d.home_rest)).sum()))

for lab, dd in [("ALL 2009-25", d), ("FIT 2009-21", d[d.fit]), ("TEST 2022-25", d[d.test])]:
    report_means("A. mean residual by bye configuration (%s) [home perspective]" % lab, dd, [
        ("home bye, away normal 6-8", dd.home_bye & normal(dd.away_rest)),
        ("away bye, home normal 6-8", dd.away_bye & normal(dd.home_rest)),
        ("home bye, away any non-bye", dd.home_bye & ~dd.away_bye),
        ("away bye, home any non-bye", dd.away_bye & ~dd.home_bye),
        ("both bye", dd.home_bye & dd.away_bye),
        ("no bye either side", ~dd.home_bye & ~dd.away_bye)])

# --- B. signed: bye team's residual (+ = bye team beat the line). sgn=+1 home bye, -1 away bye; exclude both-bye
one = d[d.home_bye ^ d.away_bye].copy(); one["sgn"] = np.where(one.home_bye, 1, -1)
one["nfelo_mod_for_bye_team"] = one.nfelo_bye_pts * one.sgn
print("\nB. bye team vs the line (one team on bye):")
print("  %-22s %4s | %-24s %-24s %-24s %-24s | nfelo mod | ATS bye team vs close" % ("subset", "n", "err_mkt", "err_nclose", "err_nraw", "err_rate"))
def line(lab, dd):
    cells = []
    for c in ["err_mkt", "err_nclose", "err_nraw", "err_rate"]:
        n, mu, se, p = desc(dd[c] * dd.sgn); cells.append("%+.2f (se %.2f p=%.2f)" % (mu, se, p))
    w, l, ps, pct, pv = ats_side(dd.err_mkt, dd.sgn)
    print("  %-22s %4d | %s | %+.2f     | %d-%d-%d %.3f (p=%.2f)" % (lab, len(dd), " ".join("%-24s" % c for c in cells), dd.nfelo_mod_for_bye_team.mean(), w, l, ps, pct, pv))
line("ALL", one); line("FIT", one[one.fit]); line("TEST", one[one.test])
line("ALL home-bye only", one[one.sgn == 1]); line("ALL away-bye only", one[one.sgn == -1])
line("TEST home-bye only", one[one.test & (one.sgn == 1)]); line("TEST away-bye only", one[one.test & (one.sgn == -1)])
opp_normal = pd.Series(np.where(one.home_bye, one.away_rest, one.home_rest), index=one.index).between(6, 8)
line("ALL opp normal 6-8", one[opp_normal]); line("TEST opp normal 6-8", one[one.test & opp_normal])
print("  by era:")
for lo, hi in [(2009, 2012), (2013, 2017), (2018, 2021), (2022, 2025)]:
    line("  %d-%d" % (lo, hi), one[one.season.between(lo, hi)])
# market-only larger sample (includes games without nfelo join, 2009-2015 mostly)
mo = m[m.home_bye ^ m.away_bye].copy(); mo["sgn"] = np.where(mo.home_bye, 1, -1)
n, mu, se, p = desc(mo.err_mkt * mo.sgn); w, l, ps, pct, pv = ats_side(mo.err_mkt, mo.sgn)
print("  market-only full sample n=%d: bye team vs close %+.2f (se %.2f p=%.2f); ATS %d-%d-%d %.3f (p=%.2f)" % (n, mu, se, p, w, l, ps, pct, pv))

# --- C. regression: is the bye effect distinct from generic rest differential? residual ~ bye_sgn + rest_diff_ex_bye
print("\nC. OLS residual ~ bye (home-away) + mini(6) + long(9-12) dummies (HC1), + = rested side gains")
def X_of(dd):
    return {"bye": dd.home_bye.astype(float) - dd.away_bye.astype(float),
            "mini(6)": dd.away_mini.astype(float) - dd.home_mini.astype(float),
            "long(9-12)": dd.home_long.astype(float) - dd.away_long.astype(float)}
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    X = X_of(dd); names = list(X)
    for c in ["err_mkt", "err_rate", "err_nraw"]:
        co, r = ols(dd[c], [X[k] for k in names], names)
        print("  %-4s %-9s " % (lab, c) + "  ".join("%s=%+.2f(%.2f,p=%.2f)" % (k, *co[k]) for k in names))
# --- D. market pricing of home vs away bye separately
print("\nD. market pricing: mkt_spread ~ rate_line + home_bye + away_bye (negative coef on home_bye = market moves toward home; positive on away_bye = toward away)")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    co, r = ols(dd.mkt_spread, [dd.rate_line, dd.home_bye.astype(float), dd.away_bye.astype(float)], ["rate_line", "home_bye", "away_bye"])
    print("  %-4s " % lab + "  ".join("%s=%+.3f(%.3f)" % (k, co[k][0], co[k][1]) for k in ["rate_line", "home_bye", "away_bye"]))
    print("       nfelo's own mod (pts, mean in this set): home bye %+.2f, away bye %+.2f" % (dd[dd.home_bye].nfelo_bye_pts.mean(), dd[dd.away_bye].nfelo_bye_pts.mean()))

# --- E. OOS: rating-only line + k * bye_sgn on one-bye games, test 2022-25
print("\nE. OOS 2022-25, one-team-bye games: MAE of rating-only line + k*(bye side); and nfelo raw (own asymmetric mod); and market")
t = one[one.test]
for k in [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    e = t.margin + t.rate_line - k * t.sgn
    print("  k=%.2f n=%d MAE %.3f RMSE %.3f bias(bye team) %+.2f" % (k, len(t), e.abs().mean(), np.sqrt((e**2).mean()), (e * t.sgn).mean()))
print("  nfelo raw (own mods) MAE %.3f | nfelo close MAE %.3f | market MAE %.3f" % (t.err_nraw.abs().mean(), t.err_nclose.abs().mean(), t.err_mkt.abs().mean()))
f = one[one.fit]; ks = np.arange(0, 3.01, 0.05); maes = [(f.margin + f.rate_line - k * f.sgn).abs().mean() for k in ks]
rm = [np.sqrt(((f.margin + f.rate_line - k * f.sgn) ** 2).mean()) for k in ks]
print("  fit-set n=%d: MAE-min k=%.2f, RMSE-min k=%.2f, mean gross effect (err_rate*sgn)=%+.2f" % (len(f), ks[int(np.argmin(maes))], ks[int(np.argmin(rm))], (f.err_rate * f.sgn).mean()))
# home vs away separately, fit set
for s_, lab in [(1, "home bye"), (-1, "away bye")]:
    ff = f[f.sgn == s_]; print("  fit-set %s n=%d gross effect %+.2f (se %.2f); nfelo mod %+.2f" % (lab, len(ff), (ff.err_rate * ff.sgn).mean(), (ff.err_rate * ff.sgn).std() / np.sqrt(len(ff)), ff.nfelo_mod_for_bye_team.mean()))
