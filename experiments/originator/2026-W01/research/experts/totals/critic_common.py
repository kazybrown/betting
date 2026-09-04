"""Critic helpers: build() from the expert's common.py, plus a REPAIRED nfelo join.
kit.py builds games.gid by replacing only _LAR_/_OAK_, so STL/SD ids (2009-2016) and some
2017-2019 games never match nfelo's normalized ids -> 317 REG games 2009-2019 lose Elo/QB.
fix_join() re-merges nfelo on a fully normalized id and fills the missing Elo/QB columns.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from common import build, mae, paired_mae_ci, ou_rate  # noqa
from kit import load_nfelo, norm

NCOLS = ["starting_nfelo_home", "starting_nfelo_away", "elo_dif_pts", "home_pts_vs_avg", "away_pts_vs_avg",
         "hfa_pts", "home_538_qb_adj", "away_538_qb_adj", "nfelo_home_line_close", "home_line_close", "total_line_close"]


def fix_join(m, verbose=True):
    parts = m.game_id.str.split("_", expand=True)
    m = m.copy()
    m["gid_fix"] = parts[0] + "_" + parts[1] + "_" + parts[2].map(norm) + "_" + parts[3].map(norm)
    n = load_nfelo()[["gid"] + NCOLS].rename(columns={"gid": "gid_fix"}).drop_duplicates("gid_fix")
    n = n.rename(columns={c: c + "_fx" for c in NCOLS})
    m = m.merge(n, on="gid_fix", how="left")
    before = m.loc[m.season >= 2009, "elo_sum"].notna().mean()
    for c in NCOLS:
        m[c] = m[c].fillna(m[c + "_fx"]); m.drop(columns=[c + "_fx"], inplace=True)
    m["elo_sum"] = m.home_pts_vs_avg + m.away_pts_vs_avg
    m["qb_sum"] = (m.home_538_qb_adj.fillna(0) + m.away_538_qb_adj.fillna(0)) / 25.0
    after = m.loc[m.season >= 2009, "elo_sum"].notna().mean()
    if verbose:
        print(f"[critic fix_join] elo coverage 2009+: {before:.3f} -> {after:.3f}")
    return m


def build_fixed(K_team=1, K_lg=128, verbose=False):
    m = build(K_team=K_team, K_lg=K_lg, verbose=verbose)
    m = fix_join(m, verbose=verbose)
    m["dome"] = m.is_dome.astype(int)
    m["lgppg"] = m.lg_blend / 2.0
    m["pf_dev"] = m.pf_sum - 2 * m.lgppg; m["pa_dev"] = m.pa_sum - 2 * m.lgppg
    m["gt_dev"] = m.pf_dev + m.pa_dev
    m["wind_c"] = np.where(m.outdoor == 1, m.wind_f - 8.4, 0.0)
    m["cold20"] = ((m.outdoor == 1) & (m.temp_f < 20)).astype(int)
    return m


def env_table(df, dome=2.0, cold=-1.0, precip=-1.5):
    w = df.wind_f
    tab = np.select([w <= 5, w <= 9, w <= 14, w <= 19, w <= 24], [0.5, -0.5, -1.5, -2.5, -3.5], -5.0)
    tab = np.where(df.outdoor == 1, tab + np.where(df.temp_f < 20, cold, 0.0), dome)
    tab = tab + np.where((df.outdoor == 1) & (df.precip_strict.fillna(0) == 1), precip, 0.0)
    return pd.Series(tab, index=df.index)


C_V3 = dict(elo=0.10, pf=0.34, pa=0.27, qb=0.72, div=-1.36)


def v3(x, LG=None, env=None, C=C_V3):
    LG = x.lg_blend if LG is None else LG
    env = env_table(x) if env is None else env
    return LG + C["elo"] * x.elo_sum + C["pf"] * x.pf_dev + C["pa"] * x.pa_dev + C["qb"] * x.qb_sum + C["div"] * x["div"] + env


def rep(lab, p, x, ref=None):
    ref = x.mkt_total if ref is None else ref
    dm, lo, hi, n = paired_mae_ci(p - x.total_pts, ref - x.total_pts)
    w, l, pu = ou_rate(p, x.mkt_total, x.total_pts)
    print(f"  {lab:60s} MAE={mae(p, x.total_pts):.3f} bias={(p - x.total_pts).mean():+.2f}  dMAE vs ref={dm:+.3f} [{lo:+.3f},{hi:+.3f}]  O/U {w}-{l}-{pu} ({w/max(w+l,1):.3f}) n={n}")
    return mae(p, x.total_pts)
