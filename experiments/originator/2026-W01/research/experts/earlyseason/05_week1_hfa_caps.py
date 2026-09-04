"""05 / THEORY 5 inputs: Week-1-specific HFA, the market's information beyond ratings in Week 1 (caps),
line movement open->close by week, and residual dispersion (confidence tags).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, desc, boot_ci, ols, paired_mae, BUCKETS

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna() & ~m.neutral].copy()

print("A. HFA in Week 1 vs later: intercept of margin ~ rating gap (elo_dif_pts, no HFA), non-neutral games, by era")
print("   (nfelo hfa_pts mean shown for comparison = what the engine currently uses; market-implied = mean(-mkt) - slope*mean(elo_dif))")
print("  era        wk    n | HFA est (se)   95%CI        | slope (se) | nfelo hfa_pts mean | market-implied HFA | home resid vs mkt (se)")
for era, emask in [("2009-2019", d.season.between(2009, 2019)), ("2020", d.season == 2020), ("2021-2025", d.season >= 2021), ("2022-2025", d.test), ("ALL 2009-25", d.season >= 2009)]:
    for w in ["1", "2", "3", "4", "5-9", "10+"]:
        x = d[emask & (d.wk == w)]
        if len(x) < 15: continue
        co, r = ols(x.margin, [x.elo_dif_pts], ["b"])
        hfa, se, _ = co["const"]
        # market-implied HFA: regress -mkt on elo_dif -> intercept
        cm, rm = ols(-x.mkt, [x.elo_dif_pts], ["b"])
        n_, mu, se_r, p = desc(x.err_mkt)
        print("  %-10s %-4s %4d | %+.2f (%.2f)  [%+.2f,%+.2f] | %.2f (%.2f) | %.2f | %+.2f | %+.2f (%.2f)" %
              (era, w, len(x), hfa, se, hfa - 1.96 * se, hfa + 1.96 * se, co["b"][0], co["b"][1], x.hfa_pts.mean(), cm["const"][0], mu, se_r))
# formal W1 vs W5+ HFA difference with interaction, 2009-2025 and 2021-2025
print("  Interaction test: margin ~ elo_dif + W1 + elo_dif*W1 (non-neutral):")
for lab, mask in [("2009-2025", d.season >= 2009), ("2021-2025", d.season >= 2021), ("2022-2025", d.test)]:
    x = d[mask]; w1 = (x.week == 1).astype(float)
    co, r = ols(x.margin, [x.elo_dif_pts, w1, x.elo_dif_pts * w1], ["b", "w1", "bx"])
    print("    %-9s n=%4d HFA other weeks %+.2f (se %.2f) | W1 shift %+.2f (se %.2f, p=%.3f)" % (lab, len(x), co["const"][0], co["const"][1], co["w1"][0], co["w1"][1], co["w1"][2]))
# market-only HFA (1999-2025) : home residual vs market by bucket = does the market's HFA need a W1 tweak?
print("  Market home residual by bucket 1999-2025 (non-neutral): a negative W1 value = market gives home too much in W1")
mm = m[~m.neutral]
for w in BUCKETS:
    x = mm[mm.wk == w]; n, mu, se, p = desc(x.err_mkt)
    print("    wk %-4s n=%4d home resid %+.2f (se %.2f) p=%.3f | mean market home edge (-mkt) %+.2f" % (w, n, mu, se, p, (-x.mkt).mean()))

print("\nB. CAPS: how informative is the market's deviation from the rating line in Week 1?  err_elo_line ~ (mkt - elo_line)")
print("   err_elo_line = margin + elo_line (+ = home beat the rating line); dev = mkt - elo_line (+ = market likes home LESS than ratings).")
print("   So a fully-earned market deviation gives slope -1. Reported 'info share' = -slope: 1 = fully informative (do not cap), 0 = noise.")
d["dev"] = d.mkt - d.elo_line
for lab, dd in [("ALL", d), ("FIT", d[d.fit]), ("TEST", d[d.test])]:
    for w in ["1", "2", "3", "4", "5-9", "10+"]:
        x = dd[dd.wk == w]
        co, r = ols(x.err_elo_line, [x.dev], ["s"])
        big = x[x.dev.abs() > 2.5]
        nb, mub, seb, pb = desc(-np.sign(big.dev) * big.err_elo_line)    # + = market side (vs ratings) was right by this much
        print("  %-4s wk %-4s n=%4d info share %.2f (se %.2f) | mean|dev| %.2f | share |dev|>2.5: %.2f | when |dev|>2.5 (n=%3d): market-side edge vs rating line %+.2f (se %.2f)" %
              (lab, w, len(x), -co["s"][0], co["s"][1], x.dev.abs().mean(), (x.dev.abs() > 2.5).mean(), nb, mub, seb))
print("  Interaction test: err_elo_line ~ dev + W1 + dev*W1  (is the market's information beyond ratings weaker in W1?)")
for lab, mask in [("2009-2025", d.season >= 2009), ("2015-2025", d.season >= 2015), ("FIT 2009-2021", d.fit), ("TEST 2022-2025", d.test)]:
    x = d[mask]; w1 = (x.week == 1).astype(float)
    co, r = ols(x.err_elo_line, [x.dev, w1, x.dev * w1], ["s", "w1", "sx"])
    print("    %-14s n=%4d info share other weeks %.2f (se %.2f) | W1 change %+.2f (se %.2f, p=%.3f) -> W1 info share %.2f" %
          (lab, len(x), -co["s"][0], co["s"][1], -co["sx"][0], co["sx"][1], co["sx"][2], -(co["s"][0] + co["sx"][0])))
    e4 = (x.week <= 4).astype(float)
    co, r = ols(x.err_elo_line, [x.dev, e4, x.dev * e4], ["s", "e", "sx"])
    print("    %-14s        info share W5+ %.2f (se %.2f) | W1-4 change %+.2f (se %.2f, p=%.3f) -> W1-4 info share %.2f" %
          ("", -co["s"][0], co["s"][1], -co["sx"][0], co["sx"][1], co["sx"][2], -(co["s"][0] + co["sx"][0])))
# OOS: would capping the market's deviation at +/-c (i.e. pred = elo_line + clip(dev, -c, c)) beat the market in W1? fit c on FIT
print("  OOS cap test on TEST 2022-25: pred = elo_line + clip(mkt - elo_line, -c, c); MAE vs market (paired)")
for w in ["1", "1-4", "5+"]:
    mask_f = d.fit & ((d.week == 1) if w == "1" else (d.week <= 4) if w == "1-4" else (d.week >= 5))
    mask_t = d.test & ((d.week == 1) if w == "1" else (d.week <= 4) if w == "1-4" else (d.week >= 5))
    xf, xt = d[mask_f], d[mask_t]
    caps = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99]
    fit_mae = {c: (xf.margin + xf.elo_line + xf.dev.clip(-c, c)).abs().mean() for c in caps}
    best = min(fit_mae, key=fit_mae.get)
    out = []
    for c in caps:
        e = xt.margin + xt.elo_line + xt.dev.clip(-c, c); dd_, lo, hi, p, n = paired_mae(e, xt.err_mkt)
        out.append("c=%s %+.3f" % (c, dd_))
    e = xt.margin + xt.elo_line + xt.dev.clip(-best, best); dd_, lo, hi, p, n = paired_mae(e, xt.err_mkt)
    print("    wk %-4s best cap on FIT = %s (FIT MAE %.3f vs uncapped %.3f) -> TEST n=%d MAE diff vs market %+.3f [%+.2f,%+.2f] p=%.2f | grid: %s" %
          (w, best, fit_mae[best], fit_mae[99], n, dd_, lo, hi, p, " ".join(out)))

print("\nC. Line movement open -> close (nfelo market columns, 2009+): spreads and totals by bucket; W1 vs W5+")
mv = m[m.home_line_open.notna() & m.home_line_close.notna()].copy()
mv["sp_move"] = (mv.home_line_close - mv.home_line_open).abs()
mv["tot_move"] = (mv.total_line_close - mv.total_line_open).abs()
for w in BUCKETS:
    x = mv[mv.wk == w]
    print("  wk %-4s n=%4d |spread move| mean %.2f  share>=1.0: %.2f  share>=2.0: %.2f | |total move| mean %.2f share>=1.0: %.2f" %
          (w, len(x), x.sp_move.mean(), (x.sp_move >= 1).mean(), (x.sp_move >= 2).mean(), x.tot_move.mean(), (x.tot_move >= 1).mean()))
a, b = mv[mv.week == 1].sp_move, mv[mv.week >= 5].sp_move
print("  W1 vs W5+ spread move: Welch p=%.3f ; totals p=%.3f" % (stats.ttest_ind(a, b, equal_var=False).pvalue, stats.ttest_ind(mv[mv.week == 1].tot_move.dropna(), mv[mv.week >= 5].tot_move.dropna(), equal_var=False).pvalue))
# does the W1 open->close move carry information (is the close better than the open in W1)?
mv["err_open"] = mv.margin + mv.home_line_open; mv["err_close"] = mv.margin + mv.home_line_close
for w in ["1", "2", "3", "4", "5-9", "10+"]:
    x = mv[mv.wk == w]; dd_, lo, hi, p, n = paired_mae(x.err_close, x.err_open)
    print("  wk %-4s close-open MAE diff %+.3f [%+.2f,%+.2f] p=%.2f (negative = close better) n=%d" % (w, dd_, lo, hi, p, n))

print("\nD. Dispersion for confidence tags: SD of residual vs market and vs rating line, and SD of (mkt - elo_line), by bucket")
for w in BUCKETS:
    x = d[d.wk == w]
    print("  wk %-4s n=%4d SD err_mkt %.2f | SD err_elo_line %.2f | SD(mkt-elo_line) %.2f | SD(mkt - nraw) %.2f | tot: SD tot_err %.2f" %
          (w, len(x), x.err_mkt.std(), x.err_elo_line.std(), (x.mkt - x.elo_line).std(), (x.mkt - x.nraw).std(), x.tot_err.std()))
