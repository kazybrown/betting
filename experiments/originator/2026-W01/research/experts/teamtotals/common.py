"""Shared loader for the team-total expert.

Conventions (README): mkt_spread = -spread_line (NEGATIVE = home favored), margin = home - away,
mkt_total = closing total. Identity under test (ORIGINATOR):
    home_tt = T/2 - S/2      away_tt = T/2 + S/2
with S in the ORIGINATOR convention. Residuals r_home = home_score - home_tt etc.
Benchmark team total = identity applied to the market CLOSING spread/total (no market team
totals exist in the kit, so this is the strongest available benchmark; it is exactly the
market's implied team total before any book shading).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kit import load_games, load_nfelo, norm, D, mae  # noqa: E402

TRAIN_MAX = 2021
TEST_SEASONS = (2022, 2023, 2024, 2025)


def load(min_season=1999, verbose=True):
    g = load_games(min_season=min_season)
    g = g[g.mkt_spread.notna() & g.mkt_total.notna() & g.result.notna()].copy()
    g["S"] = g.mkt_spread
    g["T"] = g.mkt_total
    g["home_tt"] = g["T"] / 2 - g["S"] / 2
    g["away_tt"] = g["T"] / 2 + g["S"] / 2
    g["r_home"] = g.home_score - g.home_tt
    g["r_away"] = g.away_score - g.away_tt
    # favorite / dog perspective (ties at S==0 -> treat home as 'favorite' for bookkeeping)
    g["home_fav"] = g["S"] <= 0
    g["abs_S"] = g["S"].abs()
    g["fav_score"] = np.where(g.home_fav, g.home_score, g.away_score)
    g["dog_score"] = np.where(g.home_fav, g.away_score, g.home_score)
    g["fav_tt"] = g["T"] / 2 + g.abs_S / 2
    g["dog_tt"] = g["T"] / 2 - g.abs_S / 2
    g["r_fav"] = g.fav_score - g.fav_tt
    g["r_dog"] = g.dog_score - g.dog_tt
    g["train"] = g.season <= TRAIN_MAX
    g["test"] = g.season.isin(TEST_SEASONS)
    g["post"] = g.game_type != "REG"
    if verbose:
        print(f"[common] rows={len(g)} seasons {g.season.min()}-{g.season.max()} | train={int(g.train.sum())} test={int(g.test.sum())}")
        print(f"[common] sign check corr(S, margin) = {np.corrcoef(g.S, g.margin)[0,1]:.3f} (must be strongly negative); "
              f"corr(T, total_pts) = {np.corrcoef(g['T'], g.total_pts)[0,1]:.3f} (positive)")
        print(f"[common] mean spread err (margin+S) = {(g.margin + g.S).mean():+.3f} | mean total err = {(g.total_pts - g['T']).mean():+.3f}")
    return g


def boot_ci(x, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se, se


def over_rate(actual, pred):
    """share of games with actual > pred, excluding pushes (actual == pred)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = a != p
    return float((a[ok] > p[ok]).mean()), int(ok.sum())
