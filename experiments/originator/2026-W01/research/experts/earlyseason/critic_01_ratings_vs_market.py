"""CRITIC 01 / Theories 1a + 1b: does the regressed preseason rating line really match the closing market in W1-4?
Attacks:
  A. Reproduce the headline W1/W10+ paired MAE numbers (elo_line & nraw vs market).
  B. Per-season W1 paired MAE differences (is the 'parity' driven by a few seasons?) + season-clustered test.
  C. PLACEBO / hindsight check: nfelo's parameters (offseason regression, k, HFA) were tuned on this same history.
     Build a NAIVE preseason rating with no hindsight: previous-season point differential per game, shrunk with a
     coefficient fitted ONLY on seasons < t (rolling origin), plus a constant HFA fitted the same way.  If the naive
     line is also ~level with the market in W1, the parity is a property of Week 1, not of nfelo's tuning.
  D. Rolling-origin 'who is right' blend (weights from seasons < t) so W1 has n=169 test games instead of 64.
  E. 1b re-checks: W1 home / dog / favourite residuals with season-clustered SE; MZ slope; W1 |line|>=10 cell.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, desc, boot_ci, paired_mae, binom, ols

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna()].copy()
print("sanity corr(mkt, margin) = %.3f (must be negative)" % np.corrcoef(m.mkt, m.margin)[0, 1])

print("\nA. Reproduce: paired MAE(rating line) - MAE(market), 2009-2025 REG")
for w, mask in [("W1", d.week == 1), ("W2-4", d.week.between(2, 4)), ("W1-4", d.week <= 4), ("W5-9", d.week.between(5, 9)), ("W10+", d.week >= 10)]:
    x = d[mask]
    for line in ["elo_line", "nraw"]:
        dd, lo, hi, p, n = paired_mae(x[f"err_{line}"], x.err_mkt)
        print("  %-5s %-8s n=%4d MAE line %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f" % (w, line, n, x[f"err_{line}"].abs().mean(), x.err_mkt.abs().mean(), dd, lo, hi, p))

print("\nB. Per-season W1 paired diff (elo_line - market MAE) and a season-clustered test")
w1 = d[d.week == 1].copy(); w1["pd"] = w1.err_elo_line.abs() - w1.err_mkt.abs()
ps = w1.groupby("season").pd.agg(["mean", "size"])
print("  " + ", ".join("%d:%+.2f" % (s, v) for s, v in ps["mean"].items()))
print("  seasons where rating line better (diff<0): %d of %d | mean of season means %+.3f (se %.3f, t p=%.3f)" %
      (int((ps["mean"] < 0).sum()), len(ps), ps["mean"].mean(), ps["mean"].std() / np.sqrt(len(ps)), stats.ttest_1samp(ps["mean"], 0).pvalue))
r = sm.OLS(w1.pd.values, np.ones(len(w1))).fit(cov_type="cluster", cov_kwds={"groups": w1.season.values})
print("  cluster(season) mean diff %+.3f (se %.3f, p=%.3f)" % (r.params[0], r.bse[0], r.pvalues[0]))
# W10+ cluster test for the comparison the expert calls significant
w10 = d[d.week >= 10].copy(); w10["pd"] = w10.err_elo_line.abs() - w10.err_mkt.abs()
r = sm.OLS(w10.pd.values, np.ones(len(w10))).fit(cov_type="cluster", cov_kwds={"groups": w10.season.values})
print("  W10+ cluster(season) mean diff %+.3f (se %.3f, p=%.3f)" % (r.params[0], r.bse[0], r.pvalues[0]))

print("\nC. PLACEBO rating with no hindsight: previous-season avg point differential (REG), rolling-origin shrink + HFA")
# team-season previous MOV from all REG games 1999-2025
h = m[["season", "home", "margin"]].rename(columns={"home": "team"}); h["mov"] = h.margin
a = m[["season", "away", "margin"]].rename(columns={"away": "team"}); a["mov"] = -a.margin
tm = pd.concat([h[["season", "team", "mov"]], a[["season", "team", "mov"]]]).groupby(["season", "team"]).mov.mean().rename("prev_mov").reset_index()
tm["season"] = tm.season + 1
mm = m.merge(tm.rename(columns={"team": "home", "prev_mov": "home_pmov"}), on=["season", "home"], how="left")
mm = mm.merge(tm.rename(columns={"team": "away", "prev_mov": "away_pmov"}), on=["season", "away"], how="left")
mm["pmov_dif"] = mm.home_pmov - mm.away_pmov
mm = mm[mm.pmov_dif.notna() & ~mm.neutral].copy()
print("  sanity: corr(pmov_dif, margin) W1 = %.3f (positive)" % np.corrcoef(mm[mm.week == 1].pmov_dif, mm[mm.week == 1].margin)[0, 1])
rows = []
for t in range(2005, 2026):
    for wk_lab, wmask in [("W1", mm.week == 1), ("W2-4", mm.week.between(2, 4)), ("W5-9", mm.week.between(5, 9)), ("W10+", mm.week >= 10)]:
        tr = mm[(mm.season < t) & wmask]; te = mm[(mm.season == t) & wmask]
        if len(te) == 0: continue
        X = sm.add_constant(tr.pmov_dif.values); b = sm.OLS(tr.margin.values, X).fit().params
        pred_margin = b[0] + b[1] * te.pmov_dif.values
        rows.append(pd.DataFrame({"season": t, "wk": wk_lab, "err_naive": te.margin.values - pred_margin, "err_mkt": te.err_mkt.values, "b": b[1], "hfa": b[0]}))
rr = pd.concat(rows)
print("  rolling-origin 2005-2025 (fit on seasons < t): naive line = fitted HFA + b * prev-season MOV gap")
for wk_lab in ["W1", "W2-4", "W5-9", "W10+"]:
    x = rr[rr.wk == wk_lab]; dd, lo, hi, p, n = paired_mae(x.err_naive, x.err_mkt)
    print("    %-5s n=%4d MAE naive %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f | mean fitted b %.3f, HFA %.2f" %
          (wk_lab, n, x.err_naive.abs().mean(), x.err_mkt.abs().mean(), dd, lo, hi, p, x.b.mean(), x.hfa.mean()))
x = rr[(rr.wk == "W1") & (rr.season >= 2015)]; dd, lo, hi, p, n = paired_mae(x.err_naive, x.err_mkt)
print("    W1 2015-2025 only: n=%d naive %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f" % (n, x.err_naive.abs().mean(), x.err_mkt.abs().mean(), dd, lo, hi, p))
x = rr[(rr.wk == "W1") & (rr.season >= 2022)]; dd, lo, hi, p, n = paired_mae(x.err_naive, x.err_mkt)
print("    W1 2022-2025 only: n=%d naive %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f" % (n, x.err_naive.abs().mean(), x.err_mkt.abs().mean(), dd, lo, hi, p))
# same-sample comparison of naive vs nfelo elo_line in W1 (2009+)
mm9 = mm[(mm.week == 1) & mm.nfelo_dif_base.notna()].copy()
nv = rr[(rr.wk == "W1") & (rr.season >= 2009)]
print("  W1 2009-2025 same games: nfelo elo_line MAE %.3f | market %.3f (n=%d); naive rolling MAE %.3f (n=%d)" %
      (mm9.err_elo_line.abs().mean(), mm9.err_mkt.abs().mean(), len(mm9), nv.err_naive.abs().mean(), len(nv)))

print("\nD. Rolling-origin blend: margin ~ a + w_r*(-elo_line) + w_m*(-mkt) fitted on seasons < t (same bucket), applied to season t (2015-25)")
for wk_lab, wmask in [("W1", d.week == 1), ("W2-4", d.week.between(2, 4)), ("W5-9", d.week.between(5, 9)), ("W10+", d.week >= 10)]:
    eb, em, wr, wm_ = [], [], [], []
    for t in range(2015, 2026):
        tr = d[(d.season < t) & wmask]; te = d[(d.season == t) & wmask]
        X = sm.add_constant(np.column_stack([-tr.elo_line, -tr.mkt])); b = sm.OLS(tr.margin.values, X).fit().params
        pred = b[0] + b[1] * (-te.elo_line) + b[2] * (-te.mkt)
        eb.append(te.margin - pred); em.append(te.err_mkt); wr.append(b[1]); wm_.append(b[2])
    eb, em = pd.concat(eb), pd.concat(em); dd, lo, hi, p, n = paired_mae(eb, em)
    print("  %-5s n=%4d mean w_rating %.2f w_market %.2f | MAE blend %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f" % (wk_lab, n, np.mean(wr), np.mean(wm_), eb.abs().mean(), em.abs().mean(), dd, lo, hi, p))

print("\nE. Theory 1b re-checks (1999-2025 W1, n=%d): season-clustered SEs" % (m.week == 1).sum())
x = m[m.week == 1]
for lab, y in [("home residual", x.err_mkt), ("fav cover margin", (x.err_mkt * x.fav_sgn)[x.fav_sgn != 0])]:
    yy = y.values; g = x.loc[y.index, "season"].values
    r = sm.OLS(yy, np.ones(len(yy))).fit(cov_type="cluster", cov_kwds={"groups": g})
    print("  %-18s mean %+.2f  cluster se %.2f p=%.3f (n=%d)" % (lab, r.params[0], r.bse[0], r.pvalues[0], len(yy)))
# MZ slope by era with intercept
for lab, mask in [("1999-2010", x.season <= 2010), ("2011-2025", x.season >= 2011), ("ALL", x.season >= 1999)]:
    z = x[mask]; co, r = ols(z.margin, [-z.mkt], ["b"])
    print("  MZ slope %-9s %.2f (se %.2f) intercept %+.2f (se %.2f) n=%d" % (lab, co["b"][0], co["b"][1], co["const"][0], co["const"][1], len(z)))
big = x[(x.abs_line >= 10) & (x.fav_sgn != 0)]; fc = big.err_mkt * big.fav_sgn
print("  |line|>=10 W1 favourites: n=%d fav cover %+.2f (se %.2f) | seasons represented %d | median %+.1f" % (len(big), fc.mean(), fc.std() / np.sqrt(len(fc)), big.season.nunique(), fc.median()))
big5 = m[(m.week >= 5) & (m.abs_line >= 10) & (m.fav_sgn != 0)]; fc5 = big5.err_mkt * big5.fav_sgn
print("  |line|>=10 W5+ favourites: n=%d fav cover %+.2f (se %.2f)" % (len(big5), fc5.mean(), fc5.std() / np.sqrt(len(fc5))))
# W1 big favourites by half: is it stable?
for lab, mask in [("<=2012", big.season <= 2012), (">=2013", big.season >= 2013)]:
    z = fc[mask.values]; print("    %-7s n=%2d fav cover %+.2f" % (lab, len(z), z.mean()))
