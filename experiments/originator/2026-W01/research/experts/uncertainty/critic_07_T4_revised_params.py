"""critic_07_T4_revised_params.py - the revised T4 parameters, computed (not asserted):
 5-season BASE for 2026 (spread/total), empirical k multipliers on <=2021 with a 5-season trailing base,
 OOS 2022-25 coverage/log-score of sqrt(BASE5^2 + D^2) vs sqrt(BASE3^2 + D^2) vs base-only, for spread
 (model = nfelo_b and = REAL pre-regression line where available) and total (T_elo); and the D-band table
 for the real pre-regression line so the recommended band shares are on the right engine.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_07_T4_revised_params.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

pd.set_option("display.width", 200)
m = build()
def trailing(col, s, k):
    d = m[(m.season >= s - k) & (m.season < s)]; return float(np.sqrt((d[col] ** 2).mean()))
print("BASE for 2026:", {f"spread_trail{k}": round(trailing("e_mkt", 2026, k), 3) for k in (3, 5, 8, 17)}, {f"total_trail{k}": round(trailing("e_tot", 2026, k), 3) for k in (3, 5, 8, 17)})
fit, test = m[m.era == "fit"].copy(), m[m.era == "test"].copy()
for k in (3, 5):
    for lab, ecol, dcol, bcol in [("spread", "e_nb", "D_base", "e_mkt"), ("total", "e_telo", "D_tot", "e_tot")]:
        f = fit[fit.season >= 2009 + k].copy(); f["base"] = [trailing(bcol, s, k) for s in f.season]
        z = f[ecol].abs() / np.sqrt(f.base ** 2 + f[dcol] ** 2); ks = [float(np.quantile(z, p)) for p in (0.5, 0.8, 0.9)]
        t = test.copy(); t["base"] = [trailing(bcol, s, k) for s in t.season]; sig = np.sqrt(t.base ** 2 + t[dcol] ** 2); sig0 = t.base
        cov = [float(np.mean(t[ecol].abs() <= kk * sig)) for kk in ks]; cov0 = [float(np.mean(t[ecol].abs() <= kk * sig0)) for kk in ks]
        ll, ll0 = np.mean(stats.norm.logpdf(t[ecol], 0, sig)), np.mean(stats.norm.logpdf(t[ecol], 0, sig0))
        print(f"trail{k} {lab}: k50/80/90 = {np.round(ks,3)} | OOS coverage sqrt-rule {np.round(cov,3)} base-only {np.round(cov0,3)} | logscore sqrt-rule {ll:.5f} base-only {ll0:.5f} (fit n={len(f)}, test n={len(t)})")
print("\nD-band table for the REAL nfelo pre-regression line, 2022-25 (identity excess uses in-band market RMSE):")
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
j = m.merge(h[["gid", "home_line_pre_regression"]].rename(columns={"home_line_pre_regression": "pre"}), on="gid", how="inner")
j = j[j.season >= 2022].copy(); j["D_pre"] = (j.pre - j.mkt).abs(); j["e_pre"] = j.margin + j.pre
for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3, "MED"), (3, 99, "LOW")]:
    s_ = j[(j.D_pre >= lo) & (j.D_pre < hi)]; rk, rm = np.sqrt((s_.e_mkt ** 2).mean()), np.sqrt((s_.e_pre ** 2).mean())
    per = [np.sqrt((j[(j.season == s) & (j.D_pre >= lo) & (j.D_pre < hi)].e_pre ** 2).mean()) - np.sqrt((j[(j.season == s) & (j.D_pre >= lo) & (j.D_pre < hi)].e_mkt ** 2).mean()) for s in (2022, 2023, 2024, 2025)]
    print(f"  {name}: n={len(s_)} share={len(s_)/len(j):.2f} mean D={s_.D_pre.mean():.2f} rmse mkt {rk:.2f} model {rm:.2f} excess {rm-rk:+.2f} | identity {np.sqrt(rk**2 + (s_.D_pre**2).mean()) - rk:+.2f} | excess by season {np.round(per,2)}")
print(f"  D_pre quantiles 2022-25: {np.round(np.quantile(j.D_pre, [.1,.25,.5,.75,.9]),2)} | for 55/30/15 shares the cuts would be {np.round(np.quantile(j.D_pre, [.55, .85]),2)}")
