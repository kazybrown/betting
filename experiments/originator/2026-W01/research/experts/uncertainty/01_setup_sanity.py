"""01_setup_sanity.py - build the frame, verify sign conventions, verify that nfelo_dif_base is
the pre-market-regression model number, print baselines and disagreement distributions.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/01_setup_sanity.py
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, mae, ats, FIT_MAX

m = build()
print("frame rows:", len(m), "| seasons", m.season.min(), "-", m.season.max(),
      "| REG:", (m.game_type == "REG").sum(), "| POST:", (m.game_type != "REG").sum())
print("fit (<=2021):", (m.era == "fit").sum(), "| test (2022-25):", (m.era == "test").sum())
print("total prior used for T_elo (mean realized total, fit era):", round(m.attrs["total_prior"], 3))

print("\n--- SIGN CHECKS (all should be strongly NEGATIVE) ---")
for c in ["mkt", "nfelo_c", "nfelo_o", "nfelo_b"]:
    print(f"corr({c}, margin) = {np.corrcoef(m[c], m.margin)[0,1]:.3f}")
print("corr(T_elo, total_pts) =", round(np.corrcoef(m.T_elo, m.total_pts)[0, 1], 3), "(should be positive)")
print("corr(mkt_total, total_pts) =", round(np.corrcoef(m.mkt_total, m.total_pts)[0, 1], 3))
print("mkt == nfelo home_line_close exact share:", round((m.mkt - m.home_line_close).abs().lt(0.01).mean(), 3))

print("\n--- nfelo_dif_base is pre-market? cross-check vs historic_projected_spreads.home_line_pre_regression (2021+) ---")
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
j = m.merge(h[["gid", "home_line_pre_regression", "home_dif_pre_reg", "market_regression_factor", "home_closing_line_rounded_nfelo"]], on="gid", how="inner")
print("joined:", len(j))
print("corr(nfelo_b, home_line_pre_regression) =", round(np.corrcoef(j.nfelo_b, j.home_line_pre_regression)[0, 1], 4))
print("mean |nfelo_b - home_line_pre_regression| =", round((j.nfelo_b - j.home_line_pre_regression).abs().mean(), 3))
print("corr(nfelo_dif_base, home_dif_pre_reg) =", round(np.corrcoef(j.nfelo_dif_base, j.home_dif_pre_reg)[0, 1], 4))
print("mean |nfelo_c - home_closing_line_rounded_nfelo| =", round((j.nfelo_c - j.home_closing_line_rounded_nfelo).abs().mean(), 3))
print("=> nfelo_b tracks the historic-file pre-regression line; nfelo_c tracks the regressed rounded line.")

print("\n--- BASELINE MAE (spread) fit / test ---")
for era in ["fit", "test"]:
    d = m[m.era == era]
    print(f"{era}: n={len(d)} | market {mae(-d.mkt, d.margin):.3f} | nfelo close(reg) {mae(-d.nfelo_c, d.margin):.3f} | nfelo base(unreg) {mae(-d.nfelo_b, d.margin):.3f}")
    w, l, p = ats(d.nfelo_b, d.mkt, d.margin); print(f"   nfelo base vs market ATS {w}-{l}-{p} ({w/(w+l):.3f})")
    w, l, p = ats(d.nfelo_c, d.mkt, d.margin); print(f"   nfelo close vs market ATS {w}-{l}-{p} ({w/(w+l):.3f})")
print("\n--- BASELINE MAE (total) ---")
for era in ["fit", "test"]:
    d = m[(m.era == era) & m.mkt_total.notna()]
    print(f"{era}: n={len(d)} | market total {mae(d.mkt_total, d.total_pts):.3f} | T_elo {mae(d.T_elo, d.total_pts):.3f} | total SD of err_mkt {d.e_tot.std():.3f} | spread SD of err_mkt {m[m.era==era].e_mkt.std():.3f}")

print("\n--- DISAGREEMENT DISTRIBUTIONS (test era unless noted) ---")
t = m[m.era == "test"]
q = [0.1, 0.25, 0.5, 0.75, 0.9]
for c in ["D_base", "D_reg", "D_nmove", "D_mmove", "D_tot"]:
    x = t[c].dropna()
    print(f"{c:8s} n={len(x)} mean={x.mean():.2f} quantiles {dict(zip(q, np.round(np.quantile(x, q), 2)))}")
x = m.D_tmove.dropna(); print(f"D_tmove  n={len(x)} (2024-25 only) mean={x.mean():.2f} quantiles {dict(zip(q, np.round(np.quantile(x, q), 2)))}")
print("\ncorr matrix of proxies (test era):")
print(t[["D_base", "D_reg", "D_nmove", "D_mmove", "abs_mkt", "ae_mkt", "ae_nb"]].corr(method="spearman").round(3).to_string())
