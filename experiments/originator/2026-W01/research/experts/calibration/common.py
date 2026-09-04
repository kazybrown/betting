"""Shared loader for the calibration expert. Builds one frame with every line in the
ORIGINATOR convention (NEGATIVE = home favored) and margin = home - away.

Columns produced:
  mkt            market closing line = -spread_line (nflverse, README benchmark)
  mkt_nfelo      nfelo_games.home_line_close (reference only; a few sign errors)
  nfelo_close    nfelo published close (regressed ~73% toward market)
  nfelo_lin      -nfelo_dif_base/25 : the linear 25-Elo-per-point line ORIGINATOR uses
  nfelo_own      nfelo's own unregressed line reconstructed from nfelo_unregressed_se
  qbelo          FiveThirtyEight qbelo closing line (second independent engine)
  elo_dif_pts    (home Elo - away Elo)/25  (pure rating gap, no HFA/QB)
  hfa_pts        hfa_mod/25  (base HFA + surface + time-zone + div + bye mods, per nfelo)
  qb_pts         home_net_qb_mod/25
  nfelo_noqb     nfelo_lin + qb_pts  i.e. nfelo_lin with the QB adjustment removed
  err_*          margin + line  (0 = line exactly right)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kit import load_games, norm, D  # noqa: E402

TRAIN_MAX = 2021
TEST_SEASONS = (2022, 2023, 2024, 2025)


def load(verbose=True):
    n = pd.read_csv(D / "nfelo_games.csv", low_memory=False)
    s = pd.read_csv(D / "nfelo_scored_individual_games.csv", low_memory=False)
    n["season"] = n.game_id.str[:4].astype(int)
    n = n[n.season <= 2025].reset_index(drop=True)
    assert len(n) == len(s), "scored file must align row-for-row with nfelo_games (2009-2025)"
    assert (n.season.values == s.season.values).all()
    assert np.nanmax(np.abs(n.home_line_close.values - s.home_line_close.values)) == 0
    n["qbelo"] = s.qbelo_home_line_close_rounded.values
    n["nfelo_unreg_se"] = s.nfelo_unregressed_se.values
    parts = n.game_id.str.split("_", expand=True)
    n["gid"] = parts[0] + "_" + parts[1] + "_" + parts[2].map(norm) + "_" + parts[3].map(norm)

    # decomposition of nfelo's pre-regression Elo margin
    elo_dif = n.starting_nfelo_home - n.starting_nfelo_away
    comp = elo_dif + n.hfa_mod + n.home_net_qb_mod.fillna(0)
    n["decomp_ok"] = (comp - n.nfelo_dif_base).abs() < 0.01

    g = load_games()
    m = g.merge(n, on="gid", how="inner", suffixes=("", "_n"))
    m = m[m.home_line_close.notna() & m.margin.notna()].copy()
    # Benchmark = nflverse closing spread_line (README rule). nfelo's own market column
    # agrees in 92% of games within 0.5 but carries a few sign errors (e.g. 2020_01_ARI_SF).
    m["mkt"] = m.mkt_spread
    m["mkt_nfelo"] = m.home_line_close
    m["nfelo_close"] = m.nfelo_home_line_close
    m["nfelo_lin"] = -m.nfelo_dif_base / 25.0
    m["elo_dif_pts"] = (m.starting_nfelo_home - m.starting_nfelo_away) / 25.0
    m["hfa_pts"] = m.hfa_mod / 25.0
    m["qb_pts"] = m.home_net_qb_mod.fillna(0) / 25.0
    # remove the QB adjustment from the actual pre-regression line (exact for playoff rows too)
    m["nfelo_noqb"] = m.nfelo_lin + m.qb_pts
    # nfelo's own unregressed line: se gives |margin + line|; pick the root nearer nfelo_lin
    r = np.sqrt(m.nfelo_unreg_se)
    c1, c2 = -m.margin + r, -m.margin - r
    m["nfelo_own"] = np.where((c1 - m.nfelo_lin).abs() < (c2 - m.nfelo_lin).abs(), c1, c2)
    for c in ["mkt", "mkt_nfelo", "nfelo_close", "nfelo_lin", "nfelo_own", "qbelo", "nfelo_noqb"]:
        m[f"err_{c}"] = m.margin + m[c]
    m["train"] = m.season <= TRAIN_MAX
    m["test"] = m.season.isin(TEST_SEASONS)
    m["post"] = m.game_type != "REG"
    if verbose:
        print(f"[common] rows={len(m)} seasons {m.season.min()}-{m.season.max()} | REG={int((~m.post).sum())} POST={int(m.post.sum())}")
        print(f"[common] decomposition nfelo_dif_base == elo_dif + hfa_mod + qb : ok in {m.decomp_ok.mean():.3f} of rows "
              f"(fails are playoff rows: {m.loc[~m.decomp_ok,'post'].mean():.2f} share post)")
        print(f"[common] sign check corr(mkt, margin) = {np.corrcoef(m.mkt, m.margin)[0,1]:.3f} (must be strongly negative)")
        print(f"[common] MAE  mkt={m.err_mkt.abs().mean():.3f} mkt_nfelo={m.err_mkt_nfelo.abs().mean():.3f} nfelo_close={m.err_nfelo_close.abs().mean():.3f} "
              f"nfelo_lin={m.err_nfelo_lin.abs().mean():.3f} nfelo_own={m.err_nfelo_own.abs().mean():.3f} qbelo={m.err_qbelo.abs().mean():.3f}")
    return m


def boot_ci(x, stat=np.mean, B=2000, seed=0):
    """percentile bootstrap CI of stat(x)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(B, len(x)))
    v = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_mae_test(err_a, err_b):
    """MAE(a) - MAE(b) with paired bootstrap CI and a sign/t test on |a|-|b|."""
    from scipy import stats
    d = np.abs(np.asarray(err_a)) - np.abs(np.asarray(err_b))
    d = d[~np.isnan(d)]
    lo, hi = boot_ci(d)
    t = stats.ttest_1samp(d, 0)
    return float(d.mean()), lo, hi, float(t.pvalue), len(d)


if __name__ == "__main__":
    m = load()
    print(m.loc[~m.decomp_ok, ["gid", "game_type", "nfelo_dif_base", "elo_dif_pts", "hfa_pts", "qb_pts"]].head())
