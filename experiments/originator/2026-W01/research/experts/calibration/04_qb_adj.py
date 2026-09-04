"""Theory 4: does nfelo's QB adjustment (home_net_qb_mod, = home_538_qb_adj - away_538_qb_adj)
add out-of-sample accuracy beyond base Elo + HFA?"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import load, boot_ci, paired_mae_test
pd.set_option("display.width", 220)
m = load()
tr, te = m[m.train], m[m.test]
print(f"\nQB adj (points): mean|qb|={m.qb_pts.abs().mean():.2f}, sd={m.qb_pts.std():.2f}, share |qb|>=1.5: {(m.qb_pts.abs()>=1.5).mean():.3f}, >=3: {(m.qb_pts.abs()>=3).mean():.3f}")
print("mean |qb| by season:", m.groupby("season").qb_pts.apply(lambda s: s.abs().mean()).round(2).to_dict())

print("\n=== A. MAE with vs without the QB adjustment (no parameters fitted: every period is OOS w.r.t. this test) ===")
rows = []
for per, d in {"train 2009-21": tr, "test 2022-25": te, "all 2009-25": m, "test REG": te[~te.post], "2023-25 (post-538 era)": m[m.season >= 2023]}.items():
    dm, lo, hi, p, n = paired_mae_test(d.err_nfelo_lin.values, d.err_nfelo_noqb.values)
    rows.append(dict(period=per, n=n, MAE_with_qb=d.err_nfelo_lin.abs().mean(), MAE_no_qb=d.err_nfelo_noqb.abs().mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                     RMSE_with=np.sqrt((d.err_nfelo_lin**2).mean()), RMSE_no=np.sqrt((d.err_nfelo_noqb**2).mean()),
                     gap_to_mkt_with=(d.nfelo_lin-d.mkt).abs().mean(), gap_to_mkt_no=(d.nfelo_noqb-d.mkt).abs().mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== A2. Games with a large QB adjustment ===")
rows = []
for th in (1.0, 1.5, 2.0, 3.0):
    for per, d in {"test": te, "all": m}.items():
        dd = d[d.qb_pts.abs() >= th]
        dm, lo, hi, p, n = paired_mae_test(dd.err_nfelo_lin.values, dd.err_nfelo_noqb.values)
        rows.append(dict(thresh=th, period=per, n=n, MAE_with=dd.err_nfelo_lin.abs().mean(), MAE_no=dd.err_nfelo_noqb.abs().mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. Coefficient on the QB term: margin = b1*(-nfelo_noqb) + b2*qb_pts  (b2=1 means QB adj correctly scaled; 0 = worthless) ===")
rows = []
for per, (a, b_) in {"2009-13": (2009, 2013), "2014-18": (2014, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025), "train<=2021": (2009, 2021), "all": (2009, 2025)}.items():
    d = m[(m.season >= a) & (m.season <= b_)]
    X = np.column_stack([-d.nfelo_noqb.values, d.qb_pts.values])
    r = sm.OLS(d.margin.values, X).fit(cov_type="HC1")
    # does the market already price it?  err_mkt ~ qb_pts
    r2 = sm.OLS(d.err_mkt.values, sm.add_constant(d.qb_pts.values)).fit(cov_type="HC1")
    # partial: err_mkt ~ (mkt - nfelo_noqb) + qb_pts  -> does QB add info beyond base elo, given the market?
    r3 = sm.OLS(d.err_mkt.values, np.column_stack([(d.mkt - d.nfelo_noqb).values, d.qb_pts.values])).fit(cov_type="HC1")
    rows.append(dict(era=per, n=int(r.nobs), b_base=r.params[0], se_base=r.bse[0], b_qb=r.params[1], se_qb=r.bse[1], ci_qb=f"[{r.params[1]-1.96*r.bse[1]:.2f},{r.params[1]+1.96*r.bse[1]:.2f}]",
                     mkt_resid_on_qb=r2.params[1], se=r2.bse[1], p=r2.pvalues[1], partial_qb_given_mkt=r3.params[1], se3=r3.bse[1], p3=r3.pvalues[1]))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== C. Optimal QB multiplier k: line = nfelo_noqb - k*qb_pts ; fit k on train by MAE, evaluate OOS ===")
grid = np.arange(0, 2.01, 0.1)
def mae_k(d, k): return np.abs(d.margin + d.nfelo_noqb - k*d.qb_pts).mean()
k_tr = grid[int(np.argmin([mae_k(tr, k) for k in grid]))]
k_te = grid[int(np.argmin([mae_k(te, k) for k in grid]))]
print(f"train-opt k={k_tr:.1f} (train MAE {mae_k(tr, k_tr):.3f} vs k=1 {mae_k(tr, 1):.3f} vs k=0 {mae_k(tr, 0):.3f})")
print(f"TEST MAE at k_train={k_tr:.1f}: {mae_k(te, k_tr):.3f} | k=1: {mae_k(te, 1):.3f} | k=0: {mae_k(te, 0):.3f} | test-opt (in-sample) k={k_te:.1f}: {mae_k(te, k_te):.3f}")
print("test MAE grid:", {round(k, 1): round(mae_k(te, k), 3) for k in grid[::2]})
rows = []; pool = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    k = grid[int(np.argmin([mae_k(trn, kk) for kk in grid]))]
    rows.append(dict(season=t_, n=len(tst), k=k, mae_k=mae_k(tst, k), mae_1=mae_k(tst, 1), mae_0=mae_k(tst, 0)))
    pool.append(pd.DataFrame({"ek": tst.margin + tst.nfelo_noqb - k*tst.qb_pts, "e1": tst.err_nfelo_lin, "e0": tst.err_nfelo_noqb}))
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
pool = pd.concat(pool)
for a, b_ in (("e1", "e0"), ("ek", "e1")):
    dm, lo, hi, p, n = paired_mae_test(pool[a], pool[b_])
    print(f"rolling pooled {a} vs {b_}: MAE {np.abs(pool[a]).mean():.3f} vs {np.abs(pool[b_]).mean():.3f} dMAE={dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")

print("\n=== D. ATS vs market close: with vs without QB adj, and the QB-only signal ===")
def ats(d, line, th=0.0):
    k = (d[line] - d.mkt).abs() > th; dd = d[k]
    pick_home = dd[line] < dd.mkt; res = dd.margin + dd.mkt
    w = int(((pick_home & (res > 0)) | (~pick_home & (res < 0))).sum()); l = int(((pick_home & (res < 0)) | (~pick_home & (res > 0))).sum())
    ci = stats.binomtest(w, w+l).proportion_ci(0.95) if w+l else None
    return w, l, (w/(w+l) if w+l else np.nan), (f"[{ci.low:.3f},{ci.high:.3f}]" if ci else "")
for per, d in {"test 2022-25": te, "all 2009-25": m}.items():
    for line in ("nfelo_lin", "nfelo_noqb"):
        for th in (0.0, 2.0):
            w, l, pct, ci = ats(d, line, th)
            print(f"  {per:12s} {line:10s} |gap|>{th}: {w}-{l} ({pct:.3f}) {ci}")
    # QB-only signal: bet the side the QB adj favours, when |qb|>=1.5, vs the market
    dd = d[d.qb_pts.abs() >= 1.5]; pick_home = dd.qb_pts > 0; res = dd.margin + dd.mkt
    w = int(((pick_home & (res > 0)) | (~pick_home & (res < 0))).sum()); l = int(((pick_home & (res < 0)) | (~pick_home & (res > 0))).sum())
    print(f"  {per:12s} QB-side only (|qb|>=1.5) vs market: {w}-{l} ({w/(w+l):.3f}) p={stats.binomtest(w, w+l).pvalue:.3f}")
