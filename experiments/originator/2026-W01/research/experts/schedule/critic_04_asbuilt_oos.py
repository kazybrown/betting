"""CRITIC 04 - Consolidated OOS check on the ORIGINATOR proxy AS BUILT per the README
(PFF/Cole spread = -(rating dif) - (hfa_mod + tz)/25, and hfa_mod = hfa_base + tz + div + surf + home_bye + away_bye exactly, see critic_00/01).
With nfelo ratings standing in for PFF/Cole, the 54% share = -(elo + qb + hfa_mod + tz)/25 = nraw_line - tz_pts, so
   as_built = 0.46*nraw + 0.54*(nraw - tz) = nraw - 0.54*tz          (nfelo's bye mod is carried at FULL weight; tz at 1.54x)
Candidates: (S) as-built + spec table; (A) as-built, spec bye clause deleted; (B) site HFA = hfa_mod/25 (tz once) = nraw, no spec;
(C) expert's c1 = strip nfelo bye+tz, + 0.15/day rest on the blend; (D) site HFA = hfa_base/25 + div/surf (= rate_line) blended with nraw (expert's (a)).
Also: era splits of the west-team-at-1pm effect (T4a) vs market and vs ratings."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, desc, ats_side
pd.set_option("display.width", 250); rng = np.random.default_rng(5)
m = build(); d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["bye_sgn"] = d.home_bye.astype(float) - d.away_bye.astype(float)
d["short_sgn"] = d.away_short.astype(float) - d.home_short.astype(float)
d["west_early_sgn"] = ((d.away_off <= -2) & d.early & ~d.neutral).astype(float) - ((d.home_off <= -2) & d.early & (d.away_off > -2) & ~d.neutral).astype(float)
d["as_built"] = d.nraw_line - 0.54 * d.nfelo_tz_pts
lines = {"market close": d.mkt_spread,
         "(S) as-built + spec (short -0.9, bye +0.75, west -0.6)": d.as_built - 0.9 * d.short_sgn - 0.75 * d.bye_sgn - 0.6 * d.west_early_sgn,
         "(S') as-built + spec bye clause only": d.as_built - 0.75 * d.bye_sgn,
         "(A) as-built, no spec": d.as_built,
         "(B) site HFA = hfa_mod/25 (tz once) = nraw, no spec": d.nraw_line,
         "(D) expert (a): 0.46 nraw + 0.54 rate_line": 0.46 * d.nraw_line + 0.54 * d.rate_line,
         "(C) expert c1: strip bye+tz, + 0.15/day rest": 0.46 * (d.nraw_line + d.nfelo_bye_pts + d.nfelo_tz_pts) + 0.54 * d.rate_line - 0.15 * d.rd,
         "(C2) (B) with nfelo bye replaced by symmetric 1.0": d.nraw_line + d.nfelo_bye_pts - 1.0 * d.bye_sgn}
def table(lab, x):
    print("\n%s (n=%d)" % (lab, len(x)))
    print("  %-52s %7s %7s | bye games (n=%d) bias | west@1pm (n=%d) home bias | tz!=0 (n=%d) home bias" % ("line", "MAE", "RMSE", (x.bye_sgn != 0).sum(), (x.west_early_sgn == 1).sum(), (x.nfelo_tz_pts != 0).sum()))
    base = None
    for nm, ln in lines.items():
        e = x.margin + ln[x.index]; b = x.bye_sgn != 0; w = x.west_early_sgn == 1; tz = x.nfelo_tz_pts != 0
        eb = (e[b] * x.bye_sgn[b]); ew = e[w]; et = e[tz]
        extra = ""
        if nm.startswith("(S) "): base = e.abs().values
        elif base is not None:
            diff = e.abs().values - base; bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]; extra = " | vs (S): %+.4f [%+.4f, %+.4f]" % (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        print("  %-52s %7.4f %7.4f | %+.2f (se %.2f) | %+.2f (se %.2f) | %+.2f (se %.2f)%s" % (nm, e.abs().mean(), np.sqrt((e ** 2).mean()), eb.mean(), eb.std() / np.sqrt(len(eb)), ew.mean(), ew.std() / np.sqrt(len(ew)), et.mean(), et.std() / np.sqrt(len(et)), extra))
table("TEST 2022-25 (out-of-sample for every rule except that 0.15/day and 1.0 were sized on <=2021 / on the market)", d[d.test])
table("ALL 2009-25 (IN-SAMPLE, for reference only)", d)
print("\n  bye points carried for the bye team (mean over one-bye games): as-built+spec %.2f | as-built %.2f | (B) %.2f | (C) %.2f | market prices ~1.0" %
      tuple(((-(ln - d.rate_line) * d.bye_sgn)[d.bye_sgn != 0]).mean() for ln in [lines["(S) as-built + spec (short -0.9, bye +0.75, west -0.6)"], lines["(A) as-built, no spec"], lines["(B) site HFA = hfa_mod/25 (tz once) = nraw, no spec"], lines["(C) expert c1: strip bye+tz, + 0.15/day rest"]]))

print("\n=== T4a by era: MT/PT away team at 1pm ET, home residual (+ = home beat line); ATS = fade the west team ===")
x = d[(d.away_off <= -2) & d.early & ~d.neutral]
for lo, hi in [(2009, 2012), (2013, 2017), (2018, 2021), (2022, 2025)]:
    y = x[x.season.between(lo, hi)]; w, l, p_, pct, pv = ats_side(y.err_mkt, 1)
    print("  %d-%d n=%3d vs market %+.2f (se %.2f) | vs ratings %+.2f (se %.2f) | vs nfelo raw %+.2f | fade-west ATS %d-%d %.3f" % (lo, hi, len(y), *desc(y.err_mkt)[1:3], *desc(y.err_rate)[1:3], y.err_nraw.mean(), w, l, pct))
