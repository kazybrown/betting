"""Shared frame for the EARLY-SEASON expert (preseason ratings, Week 1 parameters).

Conventions (README): ORIGINATOR spread NEGATIVE = home favored; margin = home - away;
prediction error = margin + line (0 = line exactly right; POSITIVE = home beat the line).

Lines built here (all ORIGINATOR convention):
  mkt        market closing line = -spread_line (nflverse; 1999-2025)            <- BENCHMARK
  nraw       nfelo UNREGRESSED line = -nfelo_dif_base/25 (Elo dif + all site mods + 538 QB adj), 2009+
  elo_line   rating+site line WITHOUT the QB adjustment = -(elo_dif + hfa_mod)/25
             (this is what a PFF / Cole style 'rating minus rating minus HFA' path looks like)
  elo_only   -elo_dif/25  (pure preseason-rating gap, no HFA at all)
  nclose     nfelo published close (regressed ~0.7 toward the market)         reference only
Totals: mkt_total = closing total; tot_err = total_pts - mkt_total.
Week buckets: 1, 2, 3, 4, 5-9, 10+ (REG season only unless asked).
Fit = seasons <= 2021, Test = 2022-2025.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kit import load_games, load_nfelo, mae, ats  # noqa: E402

FIT_MAX = 2021
TEST = (2022, 2023, 2024, 2025)
BUCKETS = ["1", "2", "3", "4", "5-9", "10+"]


def wk_bucket(week):
    return np.select([week == 1, week == 2, week == 3, week == 4, week.between(5, 9)],
                     ["1", "2", "3", "4", "5-9"], default="10+")


def build(min_season=1999, reg_only=True):
    g = load_games(min_season=min_season)
    n = load_nfelo()
    keep = ["gid", "starting_nfelo_home", "starting_nfelo_away", "hfa_mod", "hfa_base_mod",
            "home_net_qb_mod", "home_538_qb_adj", "away_538_qb_adj", "div_game_mod", "dif_surface_mod",
            "home_time_advantage_mod", "home_bye_mod", "away_bye_mod", "nfelo_dif_base",
            "nfelo_home_line_close", "nfelo_home_line_open", "home_line_close", "home_line_open",
            "total_line_close", "total_line_open"]
    m = g.merge(n[[k for k in keep if k in n.columns]], on="gid", how="left")
    if reg_only:
        m = m[m.game_type.eq("REG")].copy()
    m = m.dropna(subset=["mkt_spread", "margin"]).copy()
    m["mkt"] = m.mkt_spread
    m["err_mkt"] = m.margin + m.mkt
    m["tot_err"] = m.total_pts - m.mkt_total
    m["elo_dif_pts"] = (m.starting_nfelo_home - m.starting_nfelo_away) / 25.0
    m["hfa_pts"] = m.hfa_mod / 25.0
    m["qb_pts"] = m.home_net_qb_mod.fillna(0) / 25.0
    m["nraw"] = -m.nfelo_dif_base / 25.0
    m["elo_line"] = -(m.elo_dif_pts + m.hfa_pts)
    m["elo_only"] = -m.elo_dif_pts
    m["nclose"] = m.nfelo_home_line_close
    for c in ["nraw", "elo_line", "elo_only", "nclose"]:
        m[f"err_{c}"] = m.margin + m[c]
    m["home_rating"] = (m.starting_nfelo_home - 1505) / 25.0
    m["away_rating"] = (m.starting_nfelo_away - 1505) / 25.0
    m["wk"] = wk_bucket(m.week)
    m["fit"] = m.season <= FIT_MAX
    m["test"] = m.season.isin(TEST)
    m["fav_sgn"] = np.sign(-m.mkt)                     # +1 home favored, -1 away favored, 0 pick
    m["abs_line"] = m.mkt.abs()
    return m


def desc(x):
    """n, mean, se, two-sided p (t-test vs 0)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    mu = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    p = stats.ttest_1samp(x, 0).pvalue
    return n, mu, se, p


def boot_ci(x, stat=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(B, len(x)))
    v = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_mae(err_a, err_b):
    """MAE(a)-MAE(b), bootstrap 95% CI, paired t p-value, n. Negative = a better."""
    d = np.abs(np.asarray(err_a, float)) - np.abs(np.asarray(err_b, float))
    d = d[~np.isnan(d)]
    lo, hi = boot_ci(d)
    return float(d.mean()), lo, hi, float(stats.ttest_1samp(d, 0).pvalue), len(d)


def binom(w, l):
    n = w + l
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    ci = stats.binomtest(w, n).proportion_ci(0.95)
    return w / n, ci.low, ci.high, stats.binomtest(w, n, 0.5).pvalue


def ols(y, X, names, robust=True):
    import statsmodels.api as sm
    Xm = sm.add_constant(np.column_stack(X))
    r = sm.OLS(np.asarray(y, float), Xm).fit(cov_type="HC1" if robust else "nonrobust")
    out = {"const": (r.params[0], r.bse[0], r.pvalues[0])}
    for i, nm in enumerate(names):
        out[nm] = (r.params[i + 1], r.bse[i + 1], r.pvalues[i + 1])
    return out, r
