"""02_theory1_disagreement_vs_abserr.py - THEORY 1: does model/market disagreement predict the
size of the error of (a) the market number and (b) the nfelo number?
  |err| ~ a + b*D fitted on seasons <= 2021, evaluated on 2022-2025 (slope re-estimated OOS with
  HC1 SEs, Spearman rho, tercile calibration with bootstrap CIs, OOS skill of the fitted line vs a
  constant). Repeated for D_base (unregressed nfelo vs market), D_reg (regressed nfelo vs market),
  D_nmove (nfelo open->close), D_mmove (market open->close), and for totals D_tot / D_tmove.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/02_theory1_disagreement_vs_abserr.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, tercile_labels, boot_mean_ci

m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
pd.set_option("display.width", 200)


def ols(y, X):
    X = sm.add_constant(np.asarray(X, float))
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="HC1")


def analyse(target, D, label, extra_ctrl=None):
    f = fit.dropna(subset=[target, D]); t = test.dropna(subset=[target, D])
    r_fit = ols(f[target], f[D])
    r_test = ols(t[target], t[D])
    rho, p_rho = stats.spearmanr(t[D], t[target])
    # OOS skill: predict |err| in test with fit-era line vs fit-era constant
    pred_line = r_fit.params[0] + r_fit.params[1] * t[D]
    pred_const = f[target].mean()
    mae_line = np.abs(t[target] - pred_line).mean(); mae_const = np.abs(t[target] - pred_const).mean()
    # terciles with fit-era edges
    lab, edges = tercile_labels(t[D].values, ref=f[D].values)
    rows = []
    for k in ["T1 low", "T2 mid", "T3 high"]:
        y = t.loc[lab == k, target]
        mu, lo, hi = boot_mean_ci(y)
        rows.append(dict(tercile=k, n=len(y), mean_abs_err=mu, ci_lo=lo, ci_hi=hi, rmse=np.sqrt((y ** 2).mean()), D_mean=t.loc[lab == k, D].mean()))
    tab = pd.DataFrame(rows).set_index("tercile")
    # T3 vs T1 difference test (Welch)
    y1 = t.loc[lab == "T1 low", target]; y3 = t.loc[lab == "T3 high", target]
    tt, pt = stats.ttest_ind(y3, y1, equal_var=False)
    ctrl = ""
    if extra_ctrl is not None:
        rc = ols(t[target], t[[D, extra_ctrl]])
        ctrl = f" | OOS slope of D controlling for {extra_ctrl}: {rc.params[1]:.3f} (p={rc.pvalues[1]:.3f})"
    print(f"\n### {label}: {target} ~ {D}")
    print(f"  FIT (<=2021, n={len(f)}): slope={r_fit.params[1]:.3f} se={r_fit.bse[1]:.3f} p={r_fit.pvalues[1]:.4f} | intercept={r_fit.params[0]:.3f}")
    print(f"  TEST (2022-25, n={len(t)}): slope={r_test.params[1]:.3f} se={r_test.bse[1]:.3f} p={r_test.pvalues[1]:.4f} | Spearman rho={rho:.3f} p={p_rho:.4f}{ctrl}")
    print(f"  OOS skill predicting |err|: MAE(fit-line)={mae_line:.4f} vs MAE(constant)={mae_const:.4f} -> improvement {100*(mae_const-mae_line)/mae_const:.2f}%")
    print(f"  tercile edges (fit era): {np.round(edges, 2)}; T3-T1 mean |err| diff = {y3.mean()-y1.mean():.3f} (Welch p={pt:.3f})")
    print(tab.round(3).to_string())
    return dict(label=label, target=target, D=D, n_test=len(t), slope_fit=r_fit.params[1], slope_test=r_test.params[1], p_test=r_test.pvalues[1], rho=rho, p_rho=p_rho, t3_minus_t1=y3.mean() - y1.mean(), p_t3t1=pt, skill_pct=100 * (mae_const - mae_line) / mae_const)


out = []
print("=" * 100); print("SPREAD: market absolute error (benchmark)")
for D in ["D_base", "D_reg", "D_nmove", "D_mmove"]:
    out.append(analyse("ae_mkt", D, "market |err|", extra_ctrl="abs_mkt"))
print("\n" + "=" * 100); print("SPREAD: nfelo UNREGRESSED (base) absolute error")
for D in ["D_base", "D_reg", "D_nmove", "D_mmove"]:
    out.append(analyse("ae_nb", D, "nfelo base |err|", extra_ctrl="abs_mkt"))
print("\n" + "=" * 100); print("SPREAD: nfelo REGRESSED (close) absolute error")
for D in ["D_base", "D_reg", "D_nmove"]:
    out.append(analyse("ae_nc", D, "nfelo close |err|", extra_ctrl="abs_mkt"))

print("\n" + "=" * 100); print("TOTALS: market total |err| and Elo-implied total |err| vs D_tot = |T_elo - mkt_total|")
out.append(analyse("ae_tot", "D_tot", "market total |err|", extra_ctrl="mkt_total"))
out.append(analyse("ae_telo", "D_tot", "T_elo total |err|", extra_ctrl="mkt_total"))
print("\n--- D_tmove (total open->close move) exists only for 2024-25: IN-SAMPLE descriptive only (n small) ---")
d = m.dropna(subset=["D_tmove", "ae_tot"])
r = ols(d.ae_tot, d.D_tmove); rho, p = stats.spearmanr(d.D_tmove, d.ae_tot)
print(f"n={len(d)} slope={r.params[1]:.3f} p={r.pvalues[1]:.3f} Spearman rho={rho:.3f} p={p:.3f}")
for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 99)]:
    y = d.loc[(d.D_tmove >= lo) & (d.D_tmove < hi), "ae_tot"]; print(f"  D_tmove in [{lo},{hi}): n={len(y)} mean|err|={y.mean():.2f} rmse={np.sqrt((y**2).mean()):.2f}")

print("\n" + "=" * 100); print("SUMMARY TABLE (test era 2022-25)")
print(pd.DataFrame(out).round(3).to_string())

print("\n--- Decomposition: why disagreement predicts model error but not market error ---")
t = test
print("test era: corr(e_mkt, nfelo_b - mkt) =", round(np.corrcoef(t.e_mkt, t.nfelo_b - t.mkt)[0, 1], 3),
      "| E[|err_model|^2] - E[|err_mkt|^2] by D_base tercile:")
lab, edges = tercile_labels(t.D_base.values, ref=fit.D_base.values)
for k in ["T1 low", "T2 mid", "T3 high"]:
    s = t[lab == k]
    print(f"  {k}: n={len(s)} rmse_mkt={np.sqrt((s.e_mkt**2).mean()):.3f} rmse_nfelo_b={np.sqrt((s.e_nb**2).mean()):.3f} rmse_nfelo_c={np.sqrt((s.e_nc**2).mean()):.3f} mean D_base={s.D_base.mean():.2f} | sqrt(rmse_mkt^2 + mean D^2)={np.sqrt((s.e_mkt**2).mean() + (s.D_base**2).mean()):.3f}")
