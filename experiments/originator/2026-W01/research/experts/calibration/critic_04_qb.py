"""CRITIC 04 (T4: nfelo QB adjustment). Attacks:
A. The expert's 'nfelo_noqb' strips the QB term from a line whose Elo ratings were maintained WITH the term, which is not a
   self-consistent no-QB model. Cleaner test: FiveThirtyEight's pure Elo (538_se) vs its qbelo (qbelo_se) -- two maintained
   models differing (mainly) by the QB adjustment. Paired |err| by period.
B. Backfill vs LIVE: historic_projected_spreads.csv carries the live home_net_qb_mod (2021-25). Compare live vs backfilled
   QB terms, and test the live pre-regression line with vs without its own live QB term (true OOS, no backfill).
C. Coefficient on the QB term inside the large-adjustment games (the backup-QB claim): err_noqb ~ const + qb_pts for |qb|>=1.5/2/3.
D. Per-season consistency of the with-vs-without dMAE (17 seasons).
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import load, paired_mae_test, D, ROOT
import sys; sys.path.insert(0, str(ROOT))
from kit import norm
pd.set_option("display.width", 220)
m = load(verbose=False); tr, te = m[m.train], m[m.test]

print("=== A. Self-consistent 538 test: pure Elo (538_se) vs qbelo (qbelo_se), MAE = mean sqrt(se) ===")
s = pd.read_csv(D / "nfelo_scored_individual_games.csv", low_memory=False)
n = pd.read_csv(D / "nfelo_games.csv", low_memory=False); n["season"] = n.game_id.str[:4].astype(int); n = n[n.season <= 2025].reset_index(drop=True)
parts = n.game_id.str.split("_", expand=True); n["gid"] = parts[0] + "_" + parts[1] + "_" + parts[2].map(norm) + "_" + parts[3].map(norm)
n["ae_538"] = np.sqrt(s["538_se"].values); n["ae_qbelo"] = np.sqrt(s["qbelo_se"].values); n["su_538"] = s["538_su"].values; n["su_qbelo"] = s["qbelo_su"].values
x = m.merge(n[["gid", "ae_538", "ae_qbelo", "su_538", "su_qbelo"]], on="gid", how="inner")
assert np.nanmax(np.abs(x.ae_qbelo - x.err_qbelo.abs())) < 1e-6
rows = []
for per, d in {"train 2009-21": x[x.train], "test 2022-25": x[x.test], "all 2009-25": x, "2023-25": x[x.season >= 2023], "2009-22 (538 live era)": x[x.season <= 2022]}.items():
    dd = d[d.ae_538.notna() & d.ae_qbelo.notna()]
    dm, lo, hi, p, nn = paired_mae_test(dd.ae_qbelo.values, dd.ae_538.values)
    rows.append(dict(period=per, n=nn, MAE_538_elo=dd.ae_538.mean(), MAE_qbelo=dd.ae_qbelo.mean(), dMAE_qbelo_minus_elo=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                     SU_538=dd.su_538.mean(), SU_qbelo=dd.su_qbelo.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# large-QB games in the 538 pair (use nfelo's qb_pts as the flag; the 538 adj is what nfelo imported pre-2023)
rows = []
for th in (1.5, 2.0, 3.0):
    for per, d in {"test": x[x.test], "all": x}.items():
        dd = d[d.qb_pts.abs() >= th]; dm, lo, hi, p, nn = paired_mae_test(dd.ae_qbelo.values, dd.ae_538.values)
        rows.append(dict(thresh=th, period=per, n=nn, MAE_538=dd.ae_538.mean(), MAE_qbelo=dd.ae_qbelo.mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. LIVE nfelo (historic file 2021-25): live QB term vs backfilled; live line with vs without its own QB term ===")
h = pd.read_csv(D / "historic_projected_spreads.csv", low_memory=False); h = h[h.season <= 2025].copy()
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
comp = (h.home_nfelo_elo - h.away_nfelo_elo) + h.home_net_HFA_mod.fillna(0) + h.home_net_bye_mod.fillna(0) + h.home_net_qb_mod.fillna(0)
h = h[np.abs(comp - h.home_dif_pre_reg) < 1]  # rows where the live decomposition holds
y = m.merge(h[["gid", "home_line_pre_regression", "home_net_qb_mod", "home_closing_line_rounded_nfelo", "nfelo_version"]].rename(columns={"home_net_qb_mod": "qb_live"}), on="gid", how="inner")
y["qb_live_pts"] = y.qb_live / 25.0; y["live"] = y.home_line_pre_regression; y["live_noqb"] = y.live + y.qb_live_pts
y["err_live"] = y.margin + y.live; y["err_live_noqb"] = y.margin + y.live_noqb
print(f"  joined n={len(y)} ({y.season.min()}-{y.season.max()}); corr(live qb, backfilled qb) = {np.corrcoef(y.qb_live_pts, y.qb_pts)[0,1]:.3f}; mean|live - backfill| = {np.abs(y.qb_live_pts - y.qb_pts).mean():.2f} pts; "
      f"mean|qb| live {y.qb_live_pts.abs().mean():.2f} vs backfill {y.qb_pts.abs().mean():.2f}")
print("  share of games where live and backfill QB terms differ by >1 pt:", round((np.abs(y.qb_live_pts - y.qb_pts) > 1).mean(), 3), "| by season:", (np.abs(y.qb_live_pts - y.qb_pts) > 1).groupby(y.season).mean().round(3).to_dict())
rows = []
for per, d in {"2021": y[y.season == 2021], "2022-25 (test)": y[y.test], "2021-25": y}.items():
    dm, lo, hi, p, nn = paired_mae_test(d.err_live.values, d.err_live_noqb.values)
    dm2, lo2, hi2, p2, _ = paired_mae_test(d.err_nfelo_lin.values, d.err_nfelo_noqb.values)
    rows.append(dict(period=per, n=nn, MAE_live_withqb=d.err_live.abs().mean(), MAE_live_noqb=d.err_live_noqb.abs().mean(), dMAE_live=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                     dMAE_backfill_same_rows=dm2, ci_backfill=f"[{lo2:.3f},{hi2:.3f}]", p_backfill=p2))
print(pd.DataFrame(rows).round(3).to_string(index=False))
rows = []
for th in (1.5, 2.0, 3.0):
    d = y[y.test & (y.qb_live_pts.abs() >= th)]; dm, lo, hi, p, nn = paired_mae_test(d.err_live.values, d.err_live_noqb.values)
    rows.append(dict(thresh=th, period="test live", n=nn, MAE_with=d.err_live.abs().mean(), MAE_no=d.err_live_noqb.abs().mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# coefficient on the live QB term
for per, d in {"2022-25 live": y[y.test], "2021-25 live": y}.items():
    r = sm.OLS(d.margin.values, np.column_stack([-d.live_noqb.values, d.qb_live_pts.values])).fit(cov_type="HC1")
    print(f"  {per}: margin ~ b1*(-live_noqb) + b2*qb_live: b1={r.params[0]:.3f}±{1.96*r.bse[0]:.3f} b2={r.params[1]:.3f}±{1.96*r.bse[1]:.3f} n={int(r.nobs)}")

print("\n=== C. Realized fraction of the QB adjustment inside large-adjustment games: err_noqb ~ const + qb_pts (slope 1 = fully realized) ===")
rows = []
for th in (0.0, 1.5, 2.0, 3.0):
    for per, d in {"train": tr, "test": te, "all": m}.items():
        dd = d[d.qb_pts.abs() >= th]
        r = sm.OLS(dd.err_nfelo_noqb.values, sm.add_constant(dd.qb_pts.values)).fit(cov_type="HC1")
        rows.append(dict(thresh=th, period=per, n=int(r.nobs), slope=r.params[1], ci=f"[{r.params[1]-1.96*r.bse[1]:.2f},{r.params[1]+1.96*r.bse[1]:.2f}]", p_vs_0=r.pvalues[1],
                         p_vs_1=float(2 * (1 - __import__('scipy').stats.norm.cdf(abs((r.params[1] - 1) / r.bse[1]))))))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== D. Per-season dMAE (with QB - without), backfilled series ===")
r = m.groupby("season").apply(lambda d: pd.Series(dict(n=len(d), dMAE=(d.err_nfelo_lin.abs() - d.err_nfelo_noqb.abs()).mean(), mean_abs_qb=d.qb_pts.abs().mean())))
print(r.round(3).to_string()); print(f"  seasons with dMAE<0 (QB helps): {(r.dMAE < 0).sum()}/17; sign test p = {__import__('scipy').stats.binomtest(int((r.dMAE < 0).sum()), 17).pvalue:.4f}")
