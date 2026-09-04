"""06: robustness of the recommended V3 formula (coefficients fixed from 05) OOS 2022-2025:
 - K_team (validation picked 1; report 1/3/6 post hoc for transparency, NOT for selection)
 - ENV table pieces: precipitation term on/off, <20F term on/off, dome level 1.5/2.0/2.5
 - league prior: lg_blend vs lg_prev vs constant 46.0 inside V3
 - post-season games (not used anywhere else): does the formula hold there?
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd
from common import build, mae, paired_mae_ci, ou_rate

C = dict(elo=0.10, pf=0.34, pa=0.27, qb=0.72, div=-1.36)


def env_table(df, dome=2.0, cold=-1.0, precip=-1.5):
    w = df.wind_f
    tab = np.select([w <= 5, w <= 9, w <= 14, w <= 19, w <= 24], [0.5, -0.5, -1.5, -2.5, -3.5], -5.0)
    tab = np.where(df.outdoor == 1, tab + np.where(df.temp_f < 20, cold, 0.0), dome)
    tab = tab + np.where((df.outdoor == 1) & (df.precip_strict.fillna(0) == 1), precip, 0.0)
    return pd.Series(tab, index=df.index)


def v3(x, LG, env):
    lgppg = x.lg_blend / 2
    return LG + C["elo"] * x.elo_sum + C["pf"] * (x.pf_sum - 2 * lgppg) + C["pa"] * (x.pa_sum - 2 * lgppg) + C["qb"] * x.qb_sum + C["div"] * x["div"] + env


def rep(lab, p, x):
    dm, lo, hi, n = paired_mae_ci(p - x.total_pts, x.mkt_total - x.total_pts)
    w, l, pu = ou_rate(p, x.mkt_total, x.total_pts)
    print(f"  {lab:48s} MAE={mae(p, x.total_pts):.3f} bias={(p - x.total_pts).mean():+.2f}  dMAE vs mkt={dm:+.3f} [{lo:+.3f},{hi:+.3f}]  O/U {w}-{l}-{pu} ({w/max(w+l,1):.3f}) n={n}")


print("== K_team sensitivity (V3, OOS 2022-2025 REG) ==")
for K in (1, 3, 6):
    m = build(K_team=K, K_lg=128, verbose=False)
    x = m[(m.game_type == "REG") & m.test & m.elo_sum.notna()].copy()
    rep(f"K_team={K}", v3(x, x.lg_blend, env_table(x)), x)

m = build(K_team=1, K_lg=128, verbose=False)
x = m[(m.game_type == "REG") & m.test & m.elo_sum.notna()].copy()
print("\n== ENV pieces (K_team=1) ==")
rep("full table (dome +2.0, <20F -1.0, precip -1.5)", v3(x, x.lg_blend, env_table(x)), x)
rep("no precipitation term", v3(x, x.lg_blend, env_table(x, precip=0.0)), x)
rep("no <20F term", v3(x, x.lg_blend, env_table(x, cold=0.0)), x)
for dm_ in (1.0, 1.5, 2.5, 3.0):
    rep(f"dome level {dm_:+.1f}", v3(x, x.lg_blend, env_table(x, dome=dm_)), x)
print("\n== league prior inside V3 ==")
rep("lg_blend (K=128)", v3(x, x.lg_blend, env_table(x)), x)
rep("lg_prev (prior season mean)", v3(x, x.lg_prev, env_table(x)), x)
rep("constant 46.0", v3(x, 46.0, env_table(x)), x)
print("\n== post-season games 2022-2025 (never used in fitting; small n) ==")
xp = m[(m.game_type != "REG") & m.test & m.elo_sum.notna()].copy()
rep("V3 on playoffs", v3(xp, xp.lg_blend, env_table(xp)), xp)
rep("spec 46+0.35elo on playoffs", 46 + 0.35 * xp.elo_sum, xp)
print(f"  playoff realized mean total={xp.total_pts.mean():.1f} market mean={xp.mkt_total.mean():.1f} n={len(xp)}")
print("\n== in-sample check (2009-2021) of the same fixed formula, for the record ==")
xi = m[(m.game_type == "REG") & m.train & m.elo_sum.notna()].copy()
rep("V3 2009-2021 (in-sample coefficients)", v3(xi, xi.lg_blend, env_table(xi)), xi)
rep("spec 46+0.35elo 2009-2021", 46 + 0.35 * xi.elo_sum, xi)
