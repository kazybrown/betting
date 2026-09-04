"""THEORY 1: is the T/2 -/+ S/2 identity the right allocation of points?

Regress home_score and away_score on (S, T) with S in ORIGINATOR convention (negative = home fav).
Identity implies home = 0.5*T - 0.5*S, away = 0.5*T + 0.5*S, intercept 0.
Also: favorite/dog perspective (fav = T/2 + |S|/2), spread-size bins, nonlinear terms,
and the OOS (2022-2025) MAE of each allocation formula fit on <= 2021.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import load, mean_ci, over_rate, mae, boot_ci

g = load(min_season=1999)
tr, te = g[g.train], g[g.test]
print(f"\nTRAIN 1999-2021 n={len(tr)}   TEST 2022-2025 n={len(te)}")

# ---------------------------------------------------------------- 1. OLS on train, HC1 SEs
def fit(df, y, X, label):
    Xm = sm.add_constant(df[X])
    r = sm.OLS(df[y], Xm).fit(cov_type="HC1")
    print(f"\n[{label}] {y} ~ const + {' + '.join(X)}   n={int(r.nobs)}")
    for k in r.params.index:
        print(f"    {k:>8s}: {r.params[k]:+.4f}  (se {r.bse[k]:.4f})")
    return r

r_h = fit(tr, "home_score", ["S", "T"], "train")
r_a = fit(tr, "away_score", ["S", "T"], "train")
# Wald tests of the identity coefficients
for r, name, hyp in [(r_h, "home", "S = -0.5, T = 0.5, const = 0"), (r_a, "away", "S = 0.5, T = 0.5, const = 0")]:
    w = r.wald_test(hyp, scalar=True)
    print(f"  Wald test identity ({name}: {hyp}): F={float(w.statistic):.2f} p={float(w.pvalue):.4f}")
    w2 = r.wald_test("S = -0.5, T = 0.5" if name == "home" else "S = 0.5, T = 0.5", scalar=True)
    print(f"  Wald test slopes only ({name}): F={float(w2.statistic):.2f} p={float(w2.pvalue):.4f}")

# favorite / dog perspective
r_f = fit(tr, "fav_score", ["abs_S", "T"], "train")
r_d = fit(tr, "dog_score", ["abs_S", "T"], "train")
for r, name, hyp in [(r_f, "fav", "abs_S = 0.5, T = 0.5, const = 0"), (r_d, "dog", "abs_S = -0.5, T = 0.5, const = 0")]:
    w = r.wald_test(hyp, scalar=True)
    print(f"  Wald test identity ({name}): F={float(w.statistic):.2f} p={float(w.pvalue):.4f}")

# ---------------------------------------------------------------- 2. residual means by spread bin (train + test separately)
bins = [-30, -13.5, -9.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 9.5, 13.5, 30]
labels = ["H fav 14+", "H fav 10-13.5", "H fav 7-9.5", "H fav 4-6.5", "H fav 1-3.5", "pick", "A fav 1-3.5", "A fav 4-6.5", "A fav 7-9.5", "A fav 10-13.5", "A fav 14+"]
g["sbin"] = pd.cut(g.S, bins=bins, labels=labels)
for nm, d in [("TRAIN 1999-2021", g[g.train]), ("TEST 2022-2025", g[g.test])]:
    print(f"\n{nm}: mean identity residual by spread bin (home_score - (T/2 - S/2); away likewise)")
    print(f"{'bin':>14s} {'n':>5s} {'r_home':>8s} {'ci':>16s} {'r_away':>8s} {'ci':>16s} {'med_h':>6s} {'med_a':>6s} {'P(h>tt)':>8s} {'P(a>tt)':>8s}")
    for b, d2 in d.groupby("sbin", observed=True):
        mh, lh, hh, _ = mean_ci(d2.r_home)
        ma, la, ha, _ = mean_ci(d2.r_away)
        oh, _ = over_rate(d2.home_score, d2.home_tt)
        oa, _ = over_rate(d2.away_score, d2.away_tt)
        print(f"{b:>14s} {len(d2):5d} {mh:+8.2f} [{lh:+6.2f},{hh:+6.2f}] {ma:+8.2f} [{la:+6.2f},{ha:+6.2f}] {d2.r_home.median():+6.2f} {d2.r_away.median():+6.2f} {oh:8.3f} {oa:8.3f}")

# favorite size bins
fbins = [-0.01, 0.5, 3, 6.5, 9.5, 13.5, 30]
flab = ["pick", "1-3", "3.5-6.5", "7-9.5", "10-13.5", "14+"]
g["fbin"] = pd.cut(g.abs_S, bins=fbins, labels=flab)
for nm, d in [("TRAIN 1999-2021", g[g.train]), ("TEST 2022-2025", g[g.test])]:
    print(f"\n{nm}: favorite / dog residuals by |S| bin (fav_score - (T/2+|S|/2), dog_score - (T/2-|S|/2))")
    print(f"{'|S|':>8s} {'n':>5s} {'r_fav':>7s} {'ci':>16s} {'r_dog':>7s} {'ci':>16s} {'r_total':>8s} {'r_margin':>9s} {'P(f>tt)':>8s} {'P(d>tt)':>8s}")
    for b, d2 in d.groupby("fbin", observed=True):
        mf, lf, hf, _ = mean_ci(d2.r_fav)
        md, ld, hd, _ = mean_ci(d2.r_dog)
        of, _ = over_rate(d2.fav_score, d2.fav_tt)
        od, _ = over_rate(d2.dog_score, d2.dog_tt)
        print(f"{b:>8s} {len(d2):5d} {mf:+7.2f} [{lf:+6.2f},{hf:+6.2f}] {md:+7.2f} [{ld:+6.2f},{hd:+6.2f}] {(d2.total_pts-d2['T']).mean():+8.2f} {(d2.margin+d2.S).mean():+9.2f} {of:8.3f} {od:8.3f}")

# ---------------------------------------------------------------- 3. OOS comparison of allocation formulas
# candidates (all fit on train only):
#   identity            : home = T/2 - S/2
#   linear OLS          : home = a + b*S + c*T (from r_h / r_a)
#   linear + |S| terms  : allow favorite/dog nonlinearity: home = a + b*S + c*T + d*|S| + e*S*|S|
#   identity + const    : home = T/2 - S/2 + k  (k = train median residual -> targets the median)
tr2, te2 = g[g.train].copy(), g[g.test].copy()
for d in (tr2, te2):
    d["S_abs"] = d.S.abs()
    d["S_sq"] = d.S * d.S_abs
    d["S2"] = d.S ** 2
    d["ST"] = d.S * d["T"]

def ols_pred(y, X):
    m = sm.OLS(tr2[y], sm.add_constant(tr2[X])).fit()
    return m.predict(sm.add_constant(te2[X])), m

results = []
for side, y, tt in [("home", "home_score", "home_tt"), ("away", "away_score", "away_tt")]:
    ident = te2[tt]
    p_lin, m_lin = ols_pred(y, ["S", "T"])
    p_nl, m_nl = ols_pred(y, ["S", "T", "S_abs", "S_sq", "ST"])
    k_med = float((tr2[y] - tr2[tt]).median())
    k_mean = float((tr2[y] - tr2[tt]).mean())
    p_ident_med = ident + k_med
    p_ident_mean = ident + k_mean
    rows = [("identity", ident), ("identity+train_mean_shift(%.2f)" % k_mean, p_ident_mean), ("identity+train_median_shift(%.2f)" % k_med, p_ident_med),
            ("OLS linear (S,T)", p_lin), ("OLS + |S|,S|S|,S*T", p_nl)]
    print(f"\nOOS 2022-2025 {side} team total, n={len(te2)}")
    print(f"{'formula':>36s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'P(over)':>8s}")
    for nm, p in rows:
        e = te2[y] - p
        o, nn = over_rate(te2[y], p)
        print(f"{nm:>36s} {e.abs().mean():7.3f} {np.sqrt((e**2).mean()):7.3f} {e.mean():+7.3f} {o:8.3f}")
        results.append((side, nm, e.abs().mean()))
    # paired bootstrap of MAE difference identity vs OLS-nonlinear
    e0 = (te2[y] - ident).abs().values
    e1 = (te2[y] - p_nl).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"   MAE(identity) - MAE(OLS nonlinear) = {np.mean(e0-e1):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
    e2 = (te2[y] - p_ident_med).abs().values
    lo, hi = boot_ci(e0 - e2)
    print(f"   MAE(identity) - MAE(identity+median shift) = {np.mean(e0-e2):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")

# ---------------------------------------------------------------- 4. era check: total-line bias by season block (drives the constant)
print("\nMean / median identity residuals by season block (home, away):")
g["block"] = pd.cut(g.season, [1998, 2004, 2010, 2016, 2021, 2025], labels=["1999-04", "2005-10", "2011-16", "2017-21", "2022-25"])
print(g.groupby("block", observed=True).agg(n=("gid", "size"), mean_r_home=("r_home", "mean"), med_r_home=("r_home", "median"),
      mean_r_away=("r_away", "mean"), med_r_away=("r_away", "median"), tot_err=("total_err_mkt", "mean"), sp_err=("spread_err_mkt", "mean")).round(2).to_string())
