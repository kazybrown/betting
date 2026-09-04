"""common.py - shared frame builder for the uncertainty / confidence-tag expert.
All lines are in ORIGINATOR convention: NEGATIVE = home favored. margin = home - away.
error = margin + line (0 = line exactly right; >0 = home beat the line).

Disagreement proxies (spread):
  D_base  = |nfelo_base - mkt_close|   nfelo_base = -(nfelo_dif_base/25): the UNREGRESSED
            nfelo number (Elo dif + HFA + QB + bye/surface mods, before market regression).
            This is the closest analog to "engine disagrees with market".
  D_reg   = |nfelo_close - mkt_close|  regressed nfelo close vs market close (the number the
            README calls "nfelo close"; it is shrunk toward the market so it is compressed).
  D_nmove = |nfelo_open - nfelo_close| nfelo open-to-close movement.
  D_mmove = |mkt_open - mkt_close|     market open-to-close movement.
Totals: no engine publishes a total. T_elo = prior + 0.35*(home_pts_vs_avg + away_pts_vs_avg)
  per the ORIGINATOR spec (prior fitted on <=2021 = mean realized total); D_tot = |T_elo - mkt_total|.
  D_tmove = |total_open - total_close| exists for 2024-25 only.
Run from /home/user/originator-2026-w01/research or with sys.path set.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, load_nfelo, mae, ats  # noqa: E402,F401

FIT_MAX = 2021          # fit seasons <= 2021
TEST_MIN = 2022         # test seasons 2022-2025


def build(reg_only=False):
    m = merged()
    n = load_nfelo()
    extra = n[["gid", "nfelo_dif_base", "nfelo_dif_open", "nfelo_dif_close"]]
    m = m.merge(extra, on="gid", how="left")
    m = m[m.mkt_spread.notna() & m.nfelo_home_line_close.notna() & m.nfelo_dif_base.notna()].copy()
    if reg_only:
        m = m[m.game_type == "REG"].copy()
    m["mkt"] = m.mkt_spread.astype(float)
    m["mkt_open"] = m.home_line_open.astype(float)
    m["nfelo_c"] = m.nfelo_home_line_close.astype(float)
    m["nfelo_o"] = m.nfelo_home_line_open.astype(float)
    m["nfelo_b"] = -(m.nfelo_dif_base / 25.0)
    # errors (0 = right)
    m["e_mkt"] = m.margin + m.mkt
    m["e_nc"] = m.margin + m.nfelo_c
    m["e_nb"] = m.margin + m.nfelo_b
    m["ae_mkt"] = m.e_mkt.abs()
    m["ae_nc"] = m.e_nc.abs()
    m["ae_nb"] = m.e_nb.abs()
    # disagreement proxies
    m["D_base"] = (m.nfelo_b - m.mkt).abs()
    m["D_reg"] = (m.nfelo_c - m.mkt).abs()
    m["D_nmove"] = (m.nfelo_o - m.nfelo_c).abs()
    m["D_mmove"] = (m.mkt_open - m.mkt).abs()
    m["sgn_dis"] = np.sign(m.nfelo_b - m.mkt)     # +: model likes home LESS than market
    m["abs_mkt"] = m.mkt.abs()
    m["era"] = np.where(m.season <= FIT_MAX, "fit", "test")
    # totals
    fitmask = m.season <= FIT_MAX
    prior = float(m.loc[fitmask & m.mkt_total.notna(), "total_pts"].mean())
    m["T_elo"] = prior + 0.35 * (m.home_pts_vs_avg + m.away_pts_vs_avg)
    m["e_tot"] = m.total_pts - m.mkt_total
    m["ae_tot"] = m.e_tot.abs()
    m["e_telo"] = m.total_pts - m.T_elo
    m["ae_telo"] = m.e_telo.abs()
    m["D_tot"] = (m.T_elo - m.mkt_total).abs()
    m["D_tmove"] = (m.total_line_open - m.total_line_close).abs()
    m["dome"] = m.is_dome.astype(int)
    m["playoff"] = (m.game_type != "REG").astype(int)
    m["wk"] = m.week.astype(int)
    m.attrs["total_prior"] = prior
    return m


def tercile_labels(x, edges=None, q=(1 / 3, 2 / 3), ref=None):
    """Label x by terciles. Edges computed on `ref` (fit era) if given, else on x."""
    src = x if ref is None else ref
    if edges is None:
        edges = np.quantile(src, q)
    lab = np.where(x <= edges[0], "T1 low", np.where(x <= edges[1], "T2 mid", "T3 high"))
    return lab, edges


def boot_mean_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
