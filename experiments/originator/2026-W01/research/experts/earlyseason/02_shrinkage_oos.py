"""02 / THEORY 2: should an originator shrink its Week-1 numbers?  Fit the shrinkage factor by week OOS.
Spread model (ORIGINATOR-style, nfelo as the only rating series we have history for):
    pred = -(k_r * elo_dif_pts + k_h * hfa_pts + k_q * qb_pts)        (k=1 everywhere = current engine)
  (a) single multiplier on the whole rating+HFA line  : pred = k * elo_line
  (b) multiplier on the rating gap only, HFA kept full : pred = -(k * elo_dif_pts + hfa_pts)
  (c) full 3-parameter fit
Totals model: implied_total = prior + c*(home_rating + away_rating), prior = previous season's realized mean total;
  engine uses c = 0.35. Fit c by bucket.
Two OOS designs: fixed split (fit <=2021, test 2022-25) and rolling-origin (fit seasons < t, test t, t=2015..2025).
Report OOS MAE/RMSE of the fitted k vs k=1 vs market, with paired bootstrap CI on the MAE difference.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, paired_mae, ols, BUCKETS, boot_ci

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna()].copy()
d["early"] = d.week <= 4
# previous-season realized mean total (league prior) from ALL scored REG games 1999+
season_mean_total = m.groupby("season").total_pts.mean()
d["prior_total"] = d.season.map(lambda s: season_mean_total.get(s - 1, np.nan))
d["rating_sum"] = d.home_rating + d.away_rating
d["tot_engine"] = d.prior_total + 0.35 * d.rating_sum       # current engine implied total (nfelo path)
d["err_tot_engine"] = d.total_pts - d.tot_engine


def fit_k_single(x):
    """MSE-optimal k for pred = k*elo_line (through origin: margin ~ -k*elo_line)."""
    z = -x.elo_line
    return float((z * x.margin).sum() / (z * z).sum())


def fit_k_rating(x):
    """pred = -(k*elo_dif + hfa): margin - hfa ~ k*elo_dif (through origin)."""
    y = x.margin - x.hfa_pts; z = x.elo_dif_pts
    return float((z * y).sum() / (z * z).sum())


def fit_3(x):
    co, r = ols(x.margin, [x.elo_dif_pts, x.hfa_pts, x.qb_pts], ["kr", "kh", "kq"], robust=True)
    return co


def se_k(x, which):
    """HC1 se of the through-origin slope."""
    import statsmodels.api as sm
    if which == "single":
        z = -x.elo_line.values; y = x.margin.values
    else:
        z = x.elo_dif_pts.values; y = (x.margin - x.hfa_pts).values
    r = sm.OLS(y, z).fit(cov_type="HC1")
    return float(r.bse[0])


print("=== SPREAD: shrinkage factor by week bucket ===")
print("A. Fixed split. k fitted on FIT (2009-2021), applied to TEST (2022-2025). MAE diff = shrunk - unshrunk (negative = shrink helps).")
print("  wk    n_fit  k_single (se)  k_rating (se) | 3-par kr  kh  kq | n_test | MAE k=1 | MAE k_single  diff [95%CI] p | MAE k_rating diff [95%CI] p | MAE mkt")
rows = []
for w in BUCKETS + ["1-4", "5+"]:
    if w == "1-4": mf = d.fit & d.early; mt = d.test & d.early
    elif w == "5+": mf = d.fit & ~d.early; mt = d.test & ~d.early
    else: mf = d.fit & (d.wk == w); mt = d.test & (d.wk == w)
    xf, xt = d[mf], d[mt]
    ks, kr = fit_k_single(xf), fit_k_rating(xf); c3 = fit_3(xf)
    p1 = xt.elo_line; ps = ks * xt.elo_line; pr = -(kr * xt.elo_dif_pts + xt.hfa_pts)
    e1, es, er = xt.margin + p1, xt.margin + ps, xt.margin + pr
    ds, lo_s, hi_s, pv_s, n = paired_mae(es, e1); dr, lo_r, hi_r, pv_r, _ = paired_mae(er, e1)
    print("  %-4s %5d   %.3f (%.3f)  %.3f (%.3f) | %.2f %.2f %.2f | %4d | %.3f | %.3f  %+.3f [%+.2f,%+.2f] %.2f | %.3f  %+.3f [%+.2f,%+.2f] %.2f | %.3f" %
          (w, len(xf), ks, se_k(xf, "single"), kr, se_k(xf, "rating"), c3["kr"][0], c3["kh"][0], c3["kq"][0], len(xt),
           e1.abs().mean(), es.abs().mean(), ds, lo_s, hi_s, pv_s, er.abs().mean(), dr, lo_r, hi_r, pv_r, xt.err_mkt.abs().mean()))

print("\nB. Rolling-origin: for each season t in 2015..2025 fit k on 2009..t-1 (same bucket), predict season t; pooled OOS.")
print("  wk    n_oos | mean k_single (range) | MAE k=1  MAE shrunk  diff [95%CI] p | RMSE k=1 RMSE shrunk | MAE mkt | k_rating: MAE diff [95%CI]")
for w in BUCKETS + ["1-4", "5+"]:
    e1s, ess, ers, mk, ks_ = [], [], [], [], []
    for t in range(2015, 2026):
        if w == "1-4": mf = (d.season < t) & d.early; mt = (d.season == t) & d.early
        elif w == "5+": mf = (d.season < t) & ~d.early; mt = (d.season == t) & ~d.early
        else: mf = (d.season < t) & (d.wk == w); mt = (d.season == t) & (d.wk == w)
        xf, xt = d[mf], d[mt]
        ks, kr = fit_k_single(xf), fit_k_rating(xf); ks_.append(ks)
        e1s.append(xt.margin + xt.elo_line); ess.append(xt.margin + ks * xt.elo_line)
        ers.append(xt.margin - (kr * xt.elo_dif_pts + xt.hfa_pts)); mk.append(xt.err_mkt)
    e1, es, er, mk = map(pd.concat, (e1s, ess, ers, mk))
    ds, lo_s, hi_s, pv_s, n = paired_mae(es, e1); dr, lo_r, hi_r, pv_r, _ = paired_mae(er, e1)
    print("  %-4s %5d | %.3f (%.2f-%.2f) | %.3f  %.3f  %+.3f [%+.2f,%+.2f] %.2f | %.3f  %.3f | %.3f | %+.3f [%+.2f,%+.2f]" %
          (w, n, np.mean(ks_), min(ks_), max(ks_), e1.abs().mean(), es.abs().mean(), ds, lo_s, hi_s, pv_s,
           np.sqrt((e1 ** 2).mean()), np.sqrt((es ** 2).mean()), mk.abs().mean(), dr, lo_r, hi_r))

print("\nC. Loss curve: OOS (2022-2025) MAE of pred = k * elo_line for a grid of k, Week 1 vs weeks 5+ (how flat is the optimum?)")
grid = np.arange(0.5, 1.51, 0.1)
for lab, mask in [("W1", d.test & (d.week == 1)), ("W1-4", d.test & d.early), ("W5+", d.test & ~d.early)]:
    x = d[mask]
    vals = [(x.margin + k * x.elo_line).abs().mean() for k in grid]
    rmse = [np.sqrt(((x.margin + k * x.elo_line) ** 2).mean()) for k in grid]
    print("  %-5s n=%4d  MAE : " % (lab, len(x)) + "  ".join("k=%.1f %.2f" % (k, v) for k, v in zip(grid, vals)))
    print("        %12s RMSE: " % "" + "  ".join("k=%.1f %.2f" % (k, v) for k, v in zip(grid, rmse)))
    print("        best k (MAE) = %.1f ; best k (RMSE) = %.1f ; MAE at k=1 minus MAE at best = %.3f" % (grid[int(np.argmin(vals))], grid[int(np.argmin(rmse))], vals[5] - min(vals)))

print("\nD. Does nfelo's own preseason regression leave the Week-1 rating gap over- or under-compressed? Slope of margin on the")
print("   pure rating gap (elo_only, no HFA) with an intercept, W1 only, by era; slope>1 means preseason Elo is too compressed.")
for lab, mask in [("2009-2015", d.season.between(2009, 2015)), ("2016-2021", d.season.between(2016, 2021)), ("2022-2025", d.test), ("ALL", d.season >= 2009)]:
    x = d[mask & (d.week == 1)]
    co, r = ols(x.margin, [x.elo_dif_pts], ["b"])
    print("  W1 %-9s n=%3d slope=%.2f (se %.2f) 95%%CI [%.2f,%.2f] | intercept (HFA at equal ratings) %+.2f (se %.2f)" %
          (lab, len(x), co["b"][0], co["b"][1], co["b"][0] - 1.96 * co["b"][1], co["b"][0] + 1.96 * co["b"][1], co["const"][0], co["const"][1]))
x = d[d.week >= 5]; co, r = ols(x.margin, [x.elo_dif_pts], ["b"])
print("  W5+ ALL      n=%4d slope=%.2f (se %.2f) | intercept %+.2f (se %.2f)" % (len(x), co["b"][0], co["b"][1], co["const"][0], co["const"][1]))

print("\n=== TOTALS: rating coefficient c in implied_total = prior + c*(home_rating+away_rating), prior = previous-season mean total ===")
print("  (prior for season s = realized mean total of season s-1 over all REG games; engine spec uses 0.35)")
print("  wk    n_fit  c_fit (se)  intercept-shift (se) | n_test | MAE engine(c=.35) | MAE c_fit  diff [95%CI] p | MAE prior-only | MAE mkt total")
for w in BUCKETS + ["1-4", "5+"]:
    if w == "1-4": mf = d.fit & d.early; mt = d.test & d.early
    elif w == "5+": mf = d.fit & ~d.early; mt = d.test & ~d.early
    else: mf = d.fit & (d.wk == w); mt = d.test & (d.wk == w)
    xf, xt = d[mf].dropna(subset=["prior_total"]), d[mt].dropna(subset=["prior_total"])
    co, r = ols(xf.total_pts - xf.prior_total, [xf.rating_sum], ["c"])
    c, a = co["c"][0], co["const"][0]
    pe = xt.prior_total + 0.35 * xt.rating_sum; pf = xt.prior_total + a + c * xt.rating_sum
    e_e, e_f, e_p, e_m = xt.total_pts - pe, xt.total_pts - pf, xt.total_pts - xt.prior_total, xt.tot_err
    dd, lo, hi, pv, n = paired_mae(e_f, e_e)
    print("  %-4s %5d  %.3f (%.3f)  %+.2f (%.2f) | %4d | %.3f | %.3f  %+.3f [%+.2f,%+.2f] %.2f | %.3f | %.3f" %
          (w, len(xf), c, co["c"][1], a, co["const"][1], len(xt), e_e.abs().mean(), e_f.abs().mean(), dd, lo, hi, pv, e_p.abs().mean(), e_m.abs().mean()))
print("  c fitted on ALL 2009-2025 by bucket (in-sample, descriptive):")
for w in BUCKETS:
    x = d[d.wk == w].dropna(subset=["prior_total"]); co, r = ols(x.total_pts - x.prior_total, [x.rating_sum], ["c"])
    print("    wk %-4s n=%4d c=%.3f (se %.3f)  shift=%+.2f (se %.2f)  R2=%.3f" % (w, len(x), co["c"][0], co["c"][1], co["const"][0], co["const"][1], r.rsquared))
