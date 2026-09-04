"""09_blend_grid_and_params.py - (a) descriptive w-grid: MAE/RMSE of mkt + w*(nfelo_b - mkt) in the
test era 2022-25 and in the fit era, to see whether ANY positive market-blend weight helps OOS;
(b) consolidated parameter table for the recommendation: trailing bases for 2026, k-multipliers,
D bands, expected excess RMSE per band, and what share of 2022-25 games each band holds.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/09_blend_grid_and_params.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

pd.set_option("display.width", 200)
m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
print("(a) blend weight grid: line = mkt + w*(nfelo_b - mkt)")
rows = []
for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
    r = dict(w=w)
    for era, d in [("fit", fit), ("test", test)]:
        e = d.margin + d.mkt + w * (d.nfelo_b - d.mkt)
        r[f"mae_{era}"] = e.abs().mean(); r[f"rmse_{era}"] = np.sqrt((e**2).mean())
    e0 = test.margin + test.mkt; e = test.margin + test.mkt + w * (test.nfelo_b - test.mkt)
    r["test_mae_diff_vs_mkt"] = e.abs().mean() - e0.abs().mean()
    if w > 0:
        r["p_paired_t"] = stats.ttest_rel(e.abs(), e0.abs()).pvalue
    rows.append(r)
print(pd.DataFrame(rows).set_index("w").round(4).to_string())

print("\n(b) parameter table")
def trailing(col, season, k=3):
    t = m[(m.season >= season - k) & (m.season < season)]; return float(np.sqrt((t[col]**2).mean()))
print(f"BASE_spread for 2026 (trailing 3 seasons 2023-25 RMSE of market-close spread error): {trailing('e_mkt', 2026):.3f}")
print(f"BASE_total  for 2026 (trailing 3 seasons 2023-25 RMSE of market-close total error):  {trailing('e_tot', 2026):.3f}")
print(f"5-season versions: spread {trailing('e_mkt', 2026, 5):.3f} | total {trailing('e_tot', 2026, 5):.3f}")
print(f"fit-era (2009-21) constants: spread {np.sqrt((fit.e_mkt**2).mean()):.3f} | total {np.sqrt((fit.e_tot**2).mean()):.3f}")
# k multipliers: |e|/sigma quantiles using constant per-season base (fit era), spread uses e_nb with sqrt rule, total e_telo
for lab, ecol, dcol, bcol in [("spread", "e_nb", "D_base", "e_mkt"), ("total", "e_telo", "D_tot", "e_tot")]:
    f = fit.copy(); f["base"] = [trailing(bcol, s) if s >= 2012 else np.sqrt((f.loc[f.season == s, bcol]**2).mean()) for s in f.season]
    z = (f[ecol].abs() / np.sqrt(f.base**2 + f[dcol]**2))
    zm = (f[bcol].abs() / f.base)
    print(f"{lab}: k_p for model number (sqrt rule) p50/p80/p90 = {np.quantile(z,.5):.3f}/{np.quantile(z,.8):.3f}/{np.quantile(z,.9):.3f} | for market number p50/p80/p90 = {np.quantile(zm,.5):.3f}/{np.quantile(zm,.8):.3f}/{np.quantile(zm,.9):.3f} | Gaussian 0.674/1.282/1.645")
print("\nexpected excess RMSE of a number sitting D points from the market close, base 12.5 (spread) / 13.1 (total):")
for D in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]:
    print(f"  D={D:>4}: spread +{np.sqrt(12.5**2 + D**2) - 12.5:.2f} | total +{np.sqrt(13.1**2 + D**2) - 13.1:.2f}")
print("\nshare of 2022-25 games by proposed band: spread D_base <1.5 / 1.5-3 / >=3:",
      [round(x, 3) for x in [(test.D_base < 1.5).mean(), ((test.D_base >= 1.5) & (test.D_base < 3)).mean(), (test.D_base >= 3).mean()]],
      "| total D_tot <2.5 / 2.5-5 / >=5:", [round(x, 3) for x in [(test.D_tot < 2.5).mean(), ((test.D_tot >= 2.5) & (test.D_tot < 5)).mean(), (test.D_tot >= 5).mean()]])
