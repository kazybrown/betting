"""CRITIC 01 (T1: 25 Elo/pt and linearity). Alternative specifications:
A. Huber (RLM) slope by era; B. per-season slopes (dispersion, share <1);
C. placebo: the same regressions on the MARKET line and nfelo's regressed close (if OLS slope>1 there too, a slope>1
   is a generic fat-tail feature of NFL margins, not evidence against 25/pt);
D. signed (home-perspective) calibration by decile, not folded by favorite;
E. REG vs POST; F. MAE-optimal scale by direct grid search (train -> test, and rolling-origin);
G. rolling-origin for the expert's best test candidate (piecewise median) and fixed 26/pt;
H. big-favorite shrink/expand test on |proj|>=10 and >=7 (paired).
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from common import load, boot_ci, paired_mae_test
pd.set_option("display.width", 220)
m = load(verbose=False)
m["proj"] = -m.nfelo_lin; m["proj_mkt"] = -m.mkt; m["proj_close"] = -m.nfelo_close; m["proj_q"] = -m.qbelo
ERAS = {"2009-13": (2009, 2013), "2014-18": (2014, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025), "train<=2021": (2009, 2021), "all": (2009, 2025)}

def slopes(d, x):
    X = d[[x]].values.astype(float); y = d.margin.values
    o = sm.OLS(y, X).fit(cov_type="HC1")
    q = QuantReg(y, sm.add_constant(X)).fit(q=0.5)
    h = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return o.params[0], o.bse[0], q.params[1], q.bse[1], h.params[0], h.bse[0]

print("=== A/C. Slopes by era for nfelo (25/pt) and PLACEBO lines (market close, nfelo regressed close, qbelo) ===")
rows = []
for era, (a, b) in ERAS.items():
    d = m[(m.season >= a) & (m.season <= b)]
    r = dict(era=era, n=len(d))
    for lab, x in [("nfelo", "proj"), ("MARKET", "proj_mkt"), ("nfelo_close", "proj_close"), ("qbelo", "proj_q")]:
        bo, so, bq, sq, bh, sh = slopes(d, x)
        r[f"{lab}_ols"] = bo; r[f"{lab}_med"] = bq; r[f"{lab}_huber"] = bh
        if lab == "nfelo": r["nfelo_ols_se"] = so; r["nfelo_med_se"] = sq; r["nfelo_huber_se"] = sh
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Reading: if the MARKET line also shows OLS slope > 1 and median slope < 1, the pattern is a property of the margin distribution, not of the Elo scale.")

print("\n=== B. Per-season slopes for nfelo proj (OLS no const; median) ===")
rows = []
for s_, d in m.groupby("season"):
    bo, so, bq, sq, bh, sh = slopes(d, "proj"); bm, _, bmq, _, _, _ = slopes(d, "proj_mkt")
    rows.append(dict(season=s_, n=len(d), b_ols=bo, b_med=bq, b_huber=bh, mkt_ols=bm, mkt_med=bmq))
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
print(f"nfelo: seasons with OLS b<1: {(r.b_ols<1).sum()}/17, median b<1: {(r.b_med<1).sum()}/17 | mean b_ols {r.b_ols.mean():.3f} sd {r.b_ols.std():.3f}; mean b_med {r.b_med.mean():.3f} sd {r.b_med.std():.3f}")
print(f"market: seasons with OLS b<1: {(r.mkt_ols<1).sum()}/17, median b<1: {(r.mkt_med<1).sum()}/17 | mean {r.mkt_ols.mean():.3f} / {r.mkt_med.mean():.3f}")

print("\n=== D. Signed calibration (home perspective): deciles of proj, mean realized margin with bootstrap CI, local slope ===")
def signed_table(d, col, label):
    d = d.copy(); d["dec"] = pd.qcut(d[col], 10, labels=False, duplicates="drop")
    out = []
    for g_, dd in d.groupby("dec"):
        lo, hi = boot_ci(dd.margin.values)
        out.append(dict(model=label, dec=g_, n=len(dd), proj_lo=dd[col].min(), proj_hi=dd[col].max(), mean_proj=dd[col].mean(), mean_real=dd.margin.mean(), ci=f"[{lo:.2f},{hi:.2f}]",
                        diff=dd.margin.mean() - dd[col].mean(), median_real=dd.margin.median()))
    return pd.DataFrame(out)
print(pd.concat([signed_table(m, "proj", "nfelo all"), signed_table(m, "proj_mkt", "market all")]).round(2).to_string(index=False))
print("-- test 2022-25 --")
print(pd.concat([signed_table(m[m.test], "proj", "nfelo test"), signed_table(m[m.test], "proj_mkt", "market test")]).round(2).to_string(index=False))

print("\n=== E. REG vs POST slope ===")
for lab, d in {"REG": m[~m.post], "POST": m[m.post]}.items():
    bo, so, bq, sq, bh, sh = slopes(d, "proj")
    print(f"  {lab:4s} n={len(d)} OLS {bo:.3f}±{1.96*so:.3f}  median {bq:.3f}±{1.96*sq:.3f}  huber {bh:.3f}±{1.96*sh:.3f}")

print("\n=== F. MAE-optimal Elo/pt by direct grid search (fit on <=2021, evaluate 2022-25; rolling-origin) ===")
grid = np.arange(20, 32.01, 0.5)
def mae_scale(d, epp): return np.abs(d.margin - d.proj * 25 / epp).mean()
tr, te = m[m.train], m[m.test]
tr_curve = {e: mae_scale(tr, e) for e in grid}; e_tr = min(tr_curve, key=tr_curve.get)
te_curve = {e: mae_scale(te, e) for e in grid}; e_te = min(te_curve, key=te_curve.get)
print(f"train-opt Elo/pt = {e_tr} (train MAE {tr_curve[e_tr]:.3f} vs 25: {tr_curve[25.0]:.3f}) -> TEST MAE {te_curve[e_tr]:.3f} vs 25: {te_curve[25.0]:.3f} | test in-sample opt {e_te} ({te_curve[e_te]:.3f})")
print("  train MAE curve:", {e: round(v, 3) for e, v in tr_curve.items() if e in (22, 23, 24, 25, 26, 27, 28, 29, 30)})
print("  test  MAE curve:", {e: round(v, 3) for e, v in te_curve.items() if e in (22, 23, 24, 25, 26, 27, 28, 29, 30)})
dm, lo, hi, p, n = paired_mae_test((te.margin - te.proj * 25 / e_tr).values, (te.margin - te.proj).values)
print(f"  paired test at train-opt {e_tr}: dMAE {dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f}")
pool = []; picks = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    cur = {e: mae_scale(trn, e) for e in grid}; eb = min(cur, key=cur.get); picks.append((t_, eb))
    pool.append(pd.DataFrame({"e_pick": tst.margin - tst.proj * 25 / eb, "e_25": tst.margin - tst.proj, "e_26": tst.margin - tst.proj * 25 / 26}))
pool = pd.concat(pool); print("  rolling picks:", picks)
for c in ("e_pick", "e_26"):
    dm, lo, hi, p, n = paired_mae_test(pool[c], pool.e_25)
    print(f"  rolling pooled {c} vs 25/pt: MAE {np.abs(pool[c]).mean():.3f} vs {np.abs(pool.e_25).mean():.3f} dMAE {dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")

print("\n=== G. Rolling-origin for the piecewise-MEDIAN candidate (expert's best OOS candidate, -0.025 p=0.105 on 2022-25) ===")
def hinges(df, col="proj"):
    s = np.sign(df[col]); a = df[col].abs()
    return np.column_stack([df[col].values, (s * np.clip(a - 7, 0, None)).values, (s * np.clip(a - 10, 0, None)).values])
pool = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    rq = QuantReg(trn.margin.values, hinges(trn)).fit(q=0.5)
    pool.append(pd.DataFrame({"e_pw": tst.margin.values - hinges(tst) @ rq.params, "e_25": tst.margin.values - tst.proj.values, "season": t_}))
pool = pd.concat(pool)
dm, lo, hi, p, n = paired_mae_test(pool.e_pw, pool.e_25)
print(f"  rolling pooled piecewise-median vs 25/pt linear: MAE {np.abs(pool.e_pw).mean():.3f} vs {np.abs(pool.e_25).mean():.3f} dMAE {dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")
print("  by season dMAE:", pool.groupby("season").apply(lambda d: (np.abs(d.e_pw) - np.abs(d.e_25)).mean()).round(3).to_dict())

print("\n=== H. Big favorites: does shrinking/expanding |proj| help? (no fitting; paired vs unscaled) ===")
rows = []
for th in (7, 10):
    for per, d in {"train": tr, "test": te, "all": m}.items():
        dd = d[d.proj.abs() >= th]
        for f in (0.85, 0.9, 1.1, 1.15):
            dm, lo, hi, p, n = paired_mae_test((dd.margin - f * dd.proj).values, (dd.margin - dd.proj).values)
            rows.append(dict(thresh=th, period=per, n=n, factor=f, MAE_base=np.abs(dd.margin - dd.proj).mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))
