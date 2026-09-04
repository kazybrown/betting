"""THEORY 3 - Rest differential (home_rest - away_rest) as a continuous effect, incl. the common asymmetries
6-day (Mon->Sun) and 10-day (Thu->Sun) rest. Fit <=2021, test 2022-25."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, report_means, desc, ols, ats_side
pd.set_option("display.width", 250)
m = build(); m = m[m.rest_valid].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["rd"] = d.rest_diff.clip(-7, 7).astype(float)
print("sample n=%d fit %d test %d | rest_diff counts:" % (len(d), d.fit.sum(), d.test.sum()), d.rest_diff.value_counts().sort_index().to_dict())

# --- A. linear slope per day of rest differential
print("\nA. OLS residual ~ rest_diff (clipped +-7); slope per day (HC1)")
for lab, dd in [("FIT", d[d.fit]), ("TEST", d[d.test]), ("ALL", d)]:
    out = []
    for c in ["err_mkt", "err_nclose", "err_nraw", "err_rate"]:
        co, r = ols(dd[c], [dd.rd], ["rd"]); out.append("%s %+.3f (se %.3f p=%.2f)" % (c, *co["rd"]))
    co, r = ols(dd.mkt_spread, [dd.rate_line, dd.rd], ["rate_line", "rd"])
    print("  %-4s n=%4d | %s | market prices %+.3f/day (se %.3f)" % (lab, len(dd), " | ".join(out), -co["rd"][0], co["rd"][1]))
# excluding byes (rest>=13 either side) so the slope is not driven by the bye dummy
nb = d[~d.home_bye & ~d.away_bye]
print("  excluding bye games:")
for lab, dd in [("FIT", nb[nb.fit]), ("TEST", nb[nb.test]), ("ALL", nb)]:
    out = []
    for c in ["err_mkt", "err_rate"]:
        co, r = ols(dd[c], [dd.rd], ["rd"]); out.append("%s %+.3f (se %.3f p=%.2f)" % (c, *co["rd"]))
    co, r = ols(dd.mkt_spread, [dd.rate_line, dd.rd], ["rate_line", "rd"])
    print("  %-4s n=%4d | %s | market prices %+.3f/day (se %.3f)" % (lab, len(dd), " | ".join(out), -co["rd"][0], co["rd"][1]))

# --- B. binned
bins = [-99, -6, -3, -1, 0, 2, 5, 99]; labels = ["<=-7", "-6..-4", "-3..-2", "-1..0", "1..2", "3..5", ">=6"]
d["bin"] = pd.cut(d.rest_diff, bins=[-99, -6.5, -3.5, -1.5, -0.5, 0.5, 2.5, 5.5, 99],
                  labels=["<=-7", "-6..-4", "-3..-2", "-1", "0", "1..2", "3..5", ">=6"])
# market movement vs rating line: residual of mkt_spread on rate_line, grouped
co, r = ols(d.mkt_spread, [d.rate_line], ["rate_line"]); d["mkt_move"] = -(d.mkt_spread - (r.params[0] + r.params[1] * d.rate_line))   # + = market favors home more than ratings
print("\nB. by rest_diff bin (ALL): mean err_mkt / err_rate / err_nraw, market move toward home vs rating line, nfelo mods")
print("  %-8s %5s | %-22s %-22s %-22s | mkt_move | nfelo bye+tz mod" % ("bin", "n", "err_mkt", "err_rate", "err_nraw"))
for b, x in d.groupby("bin", observed=True):
    cells = ["%+.2f (se %.2f)" % (desc(x[c])[1], desc(x[c])[2]) for c in ["err_mkt", "err_rate", "err_nraw"]]
    print("  %-8s %5d | %s | %+.2f    | %+.2f" % (b, len(x), " ".join("%-22s" % c for c in cells), x.mkt_move.mean(), (x.nfelo_bye_pts + x.nfelo_tz_pts).mean()))

# --- C. the common asymmetries with n>150: 6-day vs normal; 10-day vs normal; 10-day vs 6-day
normal = lambda r: r.between(7, 8)
groups = [("home 6-day (Mon->Sun), away 7-8", d.home_mini & normal(d.away_rest), -1),
          ("away 6-day, home 7-8", d.away_mini & normal(d.home_rest), +1),
          ("home 9-12 (Thu->Sun), away 7-8", d.home_long & normal(d.away_rest), +1),
          ("away 9-12, home 7-8", d.away_long & normal(d.home_rest), -1)]
print("\nC. specific asymmetries; effect shown for the MORE-RESTED side (+ = rested side beat the line)")
print("  %-34s %4s | %-22s %-22s %-22s | nfelo mod(rested) | ATS rested vs close" % ("group", "n", "err_mkt", "err_rate", "err_nraw"))
for lab, mask, sgn in groups:
    for sub, dd in [("ALL", d[mask]), ("TEST", d[mask & d.test])]:
        cells = ["%+.2f (se %.2f p=%.2f)" % desc(dd[c] * sgn)[1:] for c in ["err_mkt", "err_rate", "err_nraw"]]
        w, l, ps, pct, pv = ats_side(dd.err_mkt, sgn)
        print("  %-34s %4d | %s | %+.2f | %d-%d-%d %.3f (p=%.2f)  [%s]" % (lab if sub == "ALL" else "   (test)", len(dd), " ".join("%-22s" % c for c in cells),
              ((dd.nfelo_bye_pts + dd.nfelo_tz_pts) * sgn).mean(), w, l, ps, pct, pv, sub))
# pooled 'more-rested side' over the 6-day and 10-day asymmetries (mutually exclusive with byes)
d["asym_sgn"] = 0
d.loc[d.home_mini & normal(d.away_rest), "asym_sgn"] = -1; d.loc[d.away_mini & normal(d.home_rest), "asym_sgn"] = 1
d.loc[d.home_long & normal(d.away_rest), "asym_sgn"] = 1; d.loc[d.away_long & normal(d.home_rest), "asym_sgn"] = -1
a = d[d.asym_sgn != 0]
print("  pooled 6-day/10-day asymmetries (rested side):")
for lab, dd in [("ALL", a), ("FIT", a[a.fit]), ("TEST", a[a.test])]:
    cells = ["%+.2f (se %.2f p=%.2f)" % desc(dd[c] * dd.asym_sgn)[1:] for c in ["err_mkt", "err_rate", "err_nraw"]]
    w, l, ps, pct, pv = ats_side(dd.err_mkt, dd.asym_sgn)
    print("    %-4s n=%4d | %s | ATS %d-%d-%d %.3f (p=%.2f)" % (lab, len(dd), " ".join("%-22s" % c for c in cells), w, l, ps, pct, pv))

# --- D. OOS model comparison on the FULL test set: rating-only line + adjustment; adjustment fit on 2009-2021
print("\nD. OOS 2022-25 full sample: rating-only line + rest adjustment (fit on 2009-21). MAE / RMSE / ATS vs close (all games where adj != 0)")
f = d[d.fit]; t = d[d.test]
def fitk(f, feat):
    ks = np.arange(-0.5, 3.01, 0.05); return ks[int(np.argmin([(f.margin + f.rate_line - k * feat(f)).abs().mean() for k in ks]))]
specs = {"linear rest_diff (per day)": lambda x: x.rd,
         "bye dummy only": lambda x: x.home_bye.astype(float) - x.away_bye.astype(float),
         "bye + 6-day + 10-day dummies (one k)": lambda x: (x.home_bye.astype(float) - x.away_bye.astype(float)) + 0.25 * ((x.away_mini.astype(float) - x.home_mini.astype(float)) + (x.home_long.astype(float) - x.away_long.astype(float))),
         "sqrt-signed rest_diff": lambda x: np.sign(x.rd) * np.sqrt(np.abs(x.rd))}
base = t.margin + t.rate_line
print("  baseline rating-only: MAE %.4f RMSE %.4f | nfelo raw (own mods): MAE %.4f | market: MAE %.4f  (n=%d)" % (base.abs().mean(), np.sqrt((base**2).mean()), t.err_nraw.abs().mean(), t.err_mkt.abs().mean(), len(t)))
for lab, feat in specs.items():
    k = fitk(f, feat); e = t.margin + t.rate_line - k * feat(t); ft = feat(t) != 0
    w, l, ps, pct, pv = ats_side(t.err_mkt[ft], np.sign(feat(t)[ft]))
    print("  %-40s k=%.2f | test MAE %.4f (d=%+.4f) RMSE %.4f | games w/ adj n=%d, MAE there %.4f vs base %.4f | ATS rested side %d-%d-%d %.3f" %
          (lab, k, e.abs().mean(), e.abs().mean() - base.abs().mean(), np.sqrt((e**2).mean()), ft.sum(), e[ft].abs().mean(), base[ft].abs().mean(), w, l, ps, pct))
# paired bootstrap CI for the MAE difference of the best spec
feat = specs["bye + 6-day + 10-day dummies (one k)"]; k = fitk(f, feat); e = t.margin + t.rate_line - k * feat(t)
diff = e.abs().values - base.abs().values; rng = np.random.default_rng(0)
bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
print("  MAE diff (bye+6+10 spec vs base) = %+.4f, 95%% bootstrap CI [%+.4f, %+.4f]" % (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
