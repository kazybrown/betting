"""03_theory2_thresholds.py - THEORY 2: are the confidence-tag thresholds sensible?
  (a) unconditional 10/50/90th percentiles of |error| (market, nfelo base) in the test era;
  (b) quantile regression |err| ~ D_base at q=.1/.5/.9 fitted on <=2021, evaluated 2022-25
      (coverage of the predicted 90th percentile by D bin);
  (c) |err| distribution by D_base bins (fit and test), incl. the bins implied by the current
      SD thresholds (for two numbers SD = |diff|/sqrt(2): SD 1.2 -> |diff| 1.70, SD 2.2 -> 3.11);
  (d) same for totals with D_tot (SD 1.8 -> 2.55, SD 3.0 -> 4.24).
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/03_theory2_thresholds.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
pd.set_option("display.width", 220)
Q = [0.1, 0.5, 0.9]

print("=== (a) unconditional |error| percentiles ===")
for era, d in [("fit<=2021", fit), ("test2022-25", test)]:
    for c in ["ae_mkt", "ae_nb", "ae_nc", "ae_tot"]:
        x = d[c].dropna(); print(f"{era:12s} {c:7s} n={len(x)} p10={np.quantile(x,.1):.2f} p50={np.quantile(x,.5):.2f} p90={np.quantile(x,.9):.2f} mean={x.mean():.2f} rmse={np.sqrt((x**2).mean()):.2f}")


def bins_table(d, D, target, edges, title):
    lab = pd.cut(d[D], bins=edges, right=False, include_lowest=True)
    g = d.groupby(lab, observed=True)[target]
    tab = pd.DataFrame({"n": g.size(), "share": g.size() / len(d), "mean_abs": g.mean(), "rmse": g.apply(lambda y: np.sqrt((y**2).mean())),
                        "p10": g.quantile(.1), "p50": g.quantile(.5), "p90": g.quantile(.9)})
    print(f"\n{title}"); print(tab.round(3).to_string())
    return tab


print("\n=== (b) quantile regression |err| ~ D_base (fit <=2021), predicted quantiles at D levels ===")
for target in ["ae_mkt", "ae_nb"]:
    print(f"\n-- target {target} --")
    preds = {}
    for q in Q:
        r = smf.quantreg(f"{target} ~ D_base", fit).fit(q=q)
        preds[q] = r
        print(f"  q={q}: intercept={r.params['Intercept']:.3f} slope={r.params['D_base']:.3f} (p={r.pvalues['D_base']:.4f}) | 95% CI slope [{r.conf_int().loc['D_base',0]:.3f}, {r.conf_int().loc['D_base',1]:.3f}]")
    grid = pd.DataFrame({"D_base": [0, 0.5, 1, 1.7, 2.2, 3.1, 4, 5, 6]})
    for q in Q:
        grid[f"q{int(q*100)}"] = preds[q].predict(grid)
    print(grid.round(2).to_string(index=False))
    # OOS coverage of predicted q90 / q50 / q10 in test era by D bin
    t = test.copy()
    for q in Q:
        t[f"under_q{int(q*100)}"] = (t[target] <= preds[q].predict(t)).astype(float)
    edges = [0, 0.5, 1.0, 1.7, 2.2, 3.11, 4.5, 99]
    lab = pd.cut(t.D_base, bins=edges, right=False)
    cov = t.groupby(lab, observed=True)[[f"under_q{int(q*100)}" for q in Q]].mean(); cov["n"] = t.groupby(lab, observed=True).size()
    print("  OOS share of test games below the fit-era predicted quantile (ideal .10/.50/.90):"); print(cov.round(3).to_string())

print("\n=== (c) |err| by D_base bin, incl. bins implied by current SD thresholds (1.2 -> 1.70, 2.2 -> 3.11 on |diff|) ===")
edges = [0, 0.5, 1.0, 1.70, 2.2, 3.11, 4.5, 99]
for era, d in [("FIT<=2021", fit), ("TEST 2022-25", test)]:
    bins_table(d, "D_base", "ae_mkt", edges, f"{era}: market |err| by D_base")
    bins_table(d, "D_base", "ae_nb", edges, f"{era}: nfelo-base |err| by D_base")
print("\nCurrent 3-way tag applied to D_base with SD-equivalent cuts (HIGH <1.70, MED 1.70-3.11, LOW >3.11), TEST era:")
tag = np.where(test.D_base < 1.70, "HIGH", np.where(test.D_base <= 3.11, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    s = test[tag == k]
    print(f"  {k}: n={len(s)} share={len(s)/len(test):.2f} rmse_mkt={np.sqrt((s.e_mkt**2).mean()):.2f} rmse_nfelo_b={np.sqrt((s.e_nb**2).mean()):.2f} rmse_nfelo_c={np.sqrt((s.e_nc**2).mean()):.2f} | mean|err| mkt={s.ae_mkt.mean():.2f} nfelo_b={s.ae_nb.mean():.2f}")
print("Same with the model's literal SD thresholds applied to D_base directly (HIGH<=1.2, MED 1.2-2.2, LOW>2.2), TEST era:")
tag = np.where(test.D_base <= 1.2, "HIGH", np.where(test.D_base <= 2.2, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    s = test[tag == k]
    print(f"  {k}: n={len(s)} share={len(s)/len(test):.2f} rmse_mkt={np.sqrt((s.e_mkt**2).mean()):.2f} rmse_nfelo_b={np.sqrt((s.e_nb**2).mean()):.2f} | mean|err| mkt={s.ae_mkt.mean():.2f} nfelo_b={s.ae_nb.mean():.2f}")

print("\n=== (d) TOTALS: |err| by D_tot bin (SD 1.8 -> 2.55, 3.0 -> 4.24 on |diff|) ===")
edges_t = [0, 1, 2.55, 4.24, 6, 99]
for era, d in [("FIT<=2021", fit), ("TEST 2022-25", test)]:
    bins_table(d, "D_tot", "ae_tot", edges_t, f"{era}: market total |err| by D_tot")
    bins_table(d, "D_tot", "ae_telo", edges_t, f"{era}: T_elo total |err| by D_tot")
for target in ["ae_tot", "ae_telo"]:
    for q in Q:
        r = smf.quantreg(f"{target} ~ D_tot", fit).fit(q=q)
        print(f"  quantreg {target} q={q}: slope={r.params['D_tot']:.3f} p={r.pvalues['D_tot']:.4f}")

print("\n=== (e) what D_base level corresponds to a 10th/50th/90th-percentile market error? (test era, D_base quantiles and the |err| at those levels) ===")
t = test
for q in [0.1, 0.5, 0.9]:
    dq = np.quantile(t.D_base, q)
    near = t[(t.D_base >= dq * 0.8) & (t.D_base <= dq * 1.2 + 0.05)]
    print(f"  D_base p{int(q*100)} = {dq:.2f} -> games near that level n={len(near)}: mean|err_mkt|={near.ae_mkt.mean():.2f} mean|err_nfelo_b|={near.ae_nb.mean():.2f} rmse_mkt={np.sqrt((near.e_mkt**2).mean()):.2f}")
