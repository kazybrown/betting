"""CRITIC 02 / Theory 2: Week-1 shrinkage of the rating line.
Attacks:
  A. Reproduce the rolling-origin W1 k and the paired MAE difference.
  B. Alternative specs: (i) k fitted WITH an intercept (so HFA level and slope are separated), (ii) Huber-robust slope,
     (iii) k fitted on nraw (with QB adj), (iv) rolling-origin starting 2012 (more test seasons), (v) shrink the rating
     gap only (HFA at face value) evaluated rolling-origin with MAE and RMSE.
  C. Placebo on the naive prior-season-MOV rating: it IS shrunk (b~0.38), i.e. shrinkage is needed for an unregressed
     rating; nfelo's starting Elo is already regressed.  Quantify: SD of W1 nfelo rating vs SD of prev-season MOV/2.
  D. Totals c: rolling-origin c for W1 vs engine 0.35 and the OOS gain, separating level shift from slope.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, paired_mae, ols, boot_ci

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna()].copy()

def roll(fit_fn, pred_fn, wmask, t0=2015, t1=2025):
    e1, es = [], []; ks = []
    for t in range(t0, t1 + 1):
        xf = d[(d.season < t) & wmask]; xt = d[(d.season == t) & wmask]
        par = fit_fn(xf); ks.append(par)
        e1.append(xt.margin + xt.elo_line); es.append(xt.margin - pred_fn(xt, par))
    e1, es = pd.concat(e1), pd.concat(es)
    dd, lo, hi, p, n = paired_mae(es, e1)
    return dict(n=n, mae1=e1.abs().mean(), maes=es.abs().mean(), diff=dd, lo=lo, hi=hi, p=p, rmse1=np.sqrt((e1**2).mean()), rmses=np.sqrt((es**2).mean()), ks=ks)

print("A/B. Rolling-origin Week-1 shrinkage variants (fit seasons < t, test t), MAE diff = variant - k=1 (negative = variant better)")
specs = {
  "single k through origin (expert)": (lambda x: float(((-x.elo_line) * x.margin).sum() / ((x.elo_line) ** 2).sum()), lambda x, k: k * (-x.elo_line)),
  "k with intercept: margin ~ a + k*(-elo_line)": (lambda x: sm.OLS(x.margin.values, sm.add_constant((-x.elo_line).values)).fit().params, lambda x, p: p[0] + p[1] * (-x.elo_line)),
  "Huber-robust k with intercept": (lambda x: sm.RLM(x.margin.values, sm.add_constant((-x.elo_line).values), M=sm.robust.norms.HuberT()).fit().params, lambda x, p: p[0] + p[1] * (-x.elo_line)),
  "k on rating gap only, HFA face value": (lambda x: float((x.elo_dif_pts * (x.margin - x.hfa_pts)).sum() / (x.elo_dif_pts ** 2).sum()), lambda x, k: k * x.elo_dif_pts + x.hfa_pts),
  "single k on nraw (with QB adj)": (lambda x: float(((-x.nraw) * x.margin).sum() / (x.nraw ** 2).sum()), lambda x, k: k * (-x.nraw)),
}
for wk_lab, wmask in [("W1", d.week == 1), ("W1-4", d.week <= 4)]:
    for t0 in (2015, 2012):
        for name, (ff, pf) in specs.items():
            r = roll(ff, pf, wmask, t0=t0)
            ks = r["ks"]; kk = [k[-1] if hasattr(k, "__len__") else k for k in ks]
            print("  %-5s t0=%d %-46s n=%4d mean k %.3f (%.2f-%.2f) | MAE k=1 %.3f var %.3f diff %+.3f [%+.2f,%+.2f] p=%.2f | RMSE %.3f -> %.3f" %
                  (wk_lab, t0, name, r["n"], np.mean(kk), min(kk), max(kk), r["mae1"], r["maes"], r["diff"], r["lo"], r["hi"], r["p"], r["rmse1"], r["rmses"]))

print("\nC. Is the W1 nfelo rating already regressed?  Dispersion of ratings vs prev-season MOV (points), W1 2009-2025")
h = m[["season", "home", "margin"]].rename(columns={"home": "team"}); h["mov"] = h.margin
a = m[["season", "away", "margin"]].rename(columns={"away": "team"}); a["mov"] = -a.margin
tm = pd.concat([h[["season", "team", "mov"]], a[["season", "team", "mov"]]]).groupby(["season", "team"]).mov.mean().rename("pmov").reset_index(); tm["season"] += 1
w1 = d[d.week == 1].merge(tm.rename(columns={"team": "home", "pmov": "home_pmov"}), on=["season", "home"]).merge(tm.rename(columns={"team": "away", "pmov": "away_pmov"}), on=["season", "away"])
rat = pd.concat([w1.home_rating, w1.away_rating]); pm = pd.concat([w1.home_pmov, w1.away_pmov])
print("  SD W1 nfelo rating (pts) %.2f | SD prev-season MOV %.2f | ratio %.2f" % (rat.std(), pm.std(), rat.std() / pm.std()))
co, r = ols(w1.home_rating - w1.away_rating, [w1.home_pmov - w1.away_pmov], ["b"])
print("  nfelo W1 rating gap ~ prev-MOV gap: slope %.3f (se %.3f), R2 %.3f  -> nfelo carries ~%.0f%% of last season's MOV into W1" % (co["b"][0], co["b"][1], r.rsquared, 100 * co["b"][0]))
co, r = ols(w1.margin, [w1.home_pmov - w1.away_pmov], ["b"])
print("  margin ~ prev-MOV gap (W1): slope %.3f (se %.3f) -> the OUTCOME-optimal carry-over is ~%.0f%%" % (co["b"][0], co["b"][1], 100 * co["b"][0]))
co, r = ols(w1.margin, [w1.elo_dif_pts], ["b"])
print("  margin ~ nfelo W1 rating gap: slope %.2f (se %.2f) (1 = calibrated)" % (co["b"][0], co["b"][1]))

print("\nD. Totals: rolling-origin W1 c and level (fit seasons < t on W1 games only vs on ALL weeks), engine c=0.35, prior = prev-season mean")
smean = m.groupby("season").total_pts.mean()
d["prior_total"] = d.season.map(lambda s: smean.get(s - 1, np.nan)); d["rating_sum"] = d.home_rating + d.away_rating
dt = d.dropna(subset=["prior_total"])
for fit_scope, fmask in [("W1 only", dt.week == 1), ("all weeks", dt.week >= 1)]:
    ee, ef, el, cs = [], [], [], []
    for t in range(2015, 2026):
        xf = dt[(dt.season < t) & fmask]; xt = dt[(dt.season == t) & (dt.week == 1)]
        X = sm.add_constant(xf.rating_sum.values); b = sm.OLS((xf.total_pts - xf.prior_total).values, X).fit().params; cs.append(b[1])
        ee.append(xt.total_pts - (xt.prior_total + 0.35 * xt.rating_sum)); ef.append(xt.total_pts - (xt.prior_total + b[0] + b[1] * xt.rating_sum))
        el.append(xt.total_pts - (xt.prior_total + b[0] + 0.35 * xt.rating_sum))
    ee, ef, el = map(pd.concat, (ee, ef, el))
    d1, lo1, hi1, p1, n = paired_mae(ef, ee); d2, lo2, hi2, p2, _ = paired_mae(el, ee)
    print("  fit on %-9s n_test=%d mean c %.3f (%.2f-%.2f) | MAE engine %.3f | fitted (c+level) diff %+.3f [%+.2f,%+.2f] p=%.2f | level-only (c=.35) diff %+.3f [%+.2f,%+.2f] p=%.2f | bias engine %+.2f" %
          (fit_scope, n, np.mean(cs), min(cs), max(cs), ee.abs().mean(), d1, lo1, hi1, p1, d2, lo2, hi2, p2, ee.mean()))
