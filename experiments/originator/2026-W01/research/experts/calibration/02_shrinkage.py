"""Theory 2: how much does a market-blind origin number lose vs the close, and what
shrinkage weight w in w*engine + (1-w)*market minimizes OOS MAE (informational)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import load, boot_ci, paired_mae_test
pd.set_option("display.width", 220)
m = load()
LINES = ["mkt", "mkt_nfelo", "nfelo_close", "nfelo_lin", "nfelo_own", "qbelo"]

print("\n=== A. MAE / RMSE by period (all games incl. playoffs), paired vs market close ===")
rows = []
for per, d in {"train 2009-21": m[m.train], "test 2022-25": m[m.test], "all 2009-25": m, "test REG only": m[m.test & ~m.post]}.items():
    for c in LINES:
        e = d[f"err_{c}"].values
        dm, lo, hi, p, n = paired_mae_test(e, d.err_mkt.values)
        rows.append(dict(period=per, line=c, n=n, MAE=np.abs(e).mean(), RMSE=np.sqrt(np.nanmean(e**2)), dMAE_vs_mkt=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                         mean_abs_gap_to_mkt=np.abs(d[c]-d.mkt).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. Shrinkage weight w: pred = w*engine + (1-w)*market ===")
def best_w(d, eng, grid=np.arange(0, 1.0001, 0.05)):
    maes = [np.abs(d.margin + w*d[eng] + (1-w)*d.mkt).mean() for w in grid]
    return grid[int(np.argmin(maes))], min(maes), dict(zip(np.round(grid, 2), np.round(maes, 4)))
def reg_w(d, eng):
    # market error = -w*(engine - market)  =>  y=err_mkt, x=(mkt - eng); slope = w
    X = (d.mkt - d[eng]).values.astype(float)
    r = sm.OLS(d.err_mkt.values, X).fit(cov_type="HC1")
    return r.params[0], r.bse[0]
tr, te = m[m.train], m[m.test]
for eng in ["nfelo_lin", "nfelo_own", "qbelo", "nfelo_close"]:
    w_tr, mae_tr, _ = best_w(tr, eng)
    w_te, mae_te, grid_te = best_w(te, eng)
    mae_te_at_wtr = np.abs(te.margin + w_tr*te[eng] + (1-w_tr)*te.mkt).mean()
    bw_tr, se_tr = reg_w(tr, eng); bw_te, se_te = reg_w(te, eng); bw_all, se_all = reg_w(m, eng)
    dm, lo, hi, p, n = paired_mae_test((te.margin + w_tr*te[eng] + (1-w_tr)*te.mkt).values, te.err_mkt.values)
    print(f"{eng:12s} train-opt w={w_tr:.2f} (train MAE {mae_tr:.3f}) -> TEST MAE at w_train={mae_te_at_wtr:.3f} vs market {te.err_mkt.abs().mean():.3f} "
          f"dMAE={dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n} | test-opt w={w_te:.2f} (in-sample, MAE {mae_te:.3f}) "
          f"| OLS w: train {bw_tr:.3f}±{1.96*se_tr:.3f}, test {bw_te:.3f}±{1.96*se_te:.3f}, all {bw_all:.3f}±{1.96*se_all:.3f}")
    if eng == "nfelo_lin":
        print("   test MAE grid (nfelo_lin):", {k: v for k, v in grid_te.items() if k in (0, .1, .2, .3, .4, .5, .6, .8, 1)})

print("\n=== B2. Rolling-origin w for nfelo_lin (fit on seasons < t) ===")
rows = []; pool = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    w, _, _ = best_w(trn, "nfelo_lin", grid=np.arange(0, 1.0001, 0.02))
    e = tst.margin + w*tst.nfelo_lin + (1-w)*tst.mkt
    rows.append(dict(season=t_, n=len(tst), w=w, mae_blend=np.abs(e).mean(), mae_mkt=tst.err_mkt.abs().mean(), mae_engine=tst.err_nfelo_lin.abs().mean()))
    pool.append(pd.DataFrame({"e": e, "em": tst.err_mkt}))
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
pool = pd.concat(pool); dm, lo, hi, p, n = paired_mae_test(pool.e, pool.em)
print(f"rolling pooled blend vs market: MAE {np.abs(pool.e).mean():.3f} vs {np.abs(pool.em).mean():.3f} dMAE={dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")

print("\n=== C. Engine-only blends (no market input): nfelo_lin vs qbelo ===")
def best_w2(d, a, b, grid=np.arange(0, 1.0001, 0.05)):
    maes = [np.abs(d.margin + w*d[a] + (1-w)*d[b]).mean() for w in grid]
    return grid[int(np.argmin(maes))], min(maes)
w2, _ = best_w2(tr, "nfelo_lin", "qbelo")
e2 = te.margin + w2*te.nfelo_lin + (1-w2)*te.qbelo
for lab, e in {"nfelo_lin": te.err_nfelo_lin, "qbelo": te.err_qbelo, f"blend w_nfelo={w2:.2f} (train-opt)": e2, "50/50": te.margin + .5*te.nfelo_lin + .5*te.qbelo}.items():
    dm, lo, hi, p, n = paired_mae_test(e.values, te.err_nfelo_lin.values)
    print(f"  TEST {lab:32s} MAE={np.abs(e).mean():.3f} dMAE vs nfelo_lin={dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f}")
# also OLS of margin on both engines (train) -> weights
X = sm.add_constant(tr[["nfelo_lin", "qbelo"]].values); rr = sm.OLS(tr.margin.values, X).fit(cov_type="HC1")
print(f"  OLS(train) margin ~ nfelo_lin + qbelo: coef nfelo={rr.params[1]:.3f}±{1.96*rr.bse[1]:.3f} qbelo={rr.params[2]:.3f}±{1.96*rr.bse[2]:.3f} (sum {rr.params[1]+rr.params[2]:.3f})")

print("\n=== D. ATS of the engine side vs market close by disagreement threshold (push excluded) ===")
from scipy import stats
def ats_tab(d, eng, label):
    out = []
    for th in (0.5, 1, 2, 3, 4.5):
        k = (d[eng] - d.mkt).abs() >= th
        dd = d[k]
        pick_home = dd[eng] < dd.mkt; res = dd.margin + dd.mkt
        win = ((pick_home & (res > 0)) | (~pick_home & (res < 0))).sum(); loss = ((pick_home & (res < 0)) | (~pick_home & (res > 0))).sum()
        n = win + loss
        if n == 0: continue
        ci = stats.binomtest(int(win), int(n)).proportion_ci(0.95)
        out.append(dict(engine=label, thresh=th, n=int(n), wins=int(win), ats=win/n, ci=f"[{ci.low:.3f},{ci.high:.3f}]", p_vs_50=stats.binomtest(int(win), int(n)).pvalue,
                        p_vs_524=stats.binomtest(int(win), int(n), 0.524, alternative="greater").pvalue))
    return pd.DataFrame(out)
print("-- test 2022-25 --"); print(pd.concat([ats_tab(te, "nfelo_lin", "nfelo_lin"), ats_tab(te, "qbelo", "qbelo")]).round(3).to_string(index=False))
print("-- all 2009-25 --"); print(pd.concat([ats_tab(m, "nfelo_lin", "nfelo_lin"), ats_tab(m, "qbelo", "qbelo")]).round(3).to_string(index=False))

print("\n=== E. nfelo's own market_regression_factor (historic_projected_spreads.csv, reference only) ===")
from common import D
h = pd.read_csv(D / "historic_projected_spreads.csv", low_memory=False)
print(h.groupby("season").market_regression_factor.describe()[["count", "mean", "25%", "50%", "75%"]].round(3).to_string())
