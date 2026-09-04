"""critic_common: shared helpers for the pace-expert critique. Reads the expert's own intermediate
tables (_game_features.csv, _teamgame_feats.csv) so every re-analysis uses exactly the expert's
features; adds (i) per-season diagnostics and (ii) an alternative, leak-free league reference:
the season-to-date league mean of the raw team-game feature over games that finished on earlier
dates in the same season (fallback = the expert's prior-season / 2019 reference when < MIN_TG
team-games are available). The expert's reference lags by a full season (2019's league mean is
used for all of 2023; 2023's for 2024), which injects season-level offsets into "league-relative"
features that are not matchup information.
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experts" / "totals"))
from common import mae, paired_mae_ci, ou_rate  # noqa: F401  (the expert's own helpers)

BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "div"]
FEATS = ["spp_neut", "spp_neut_run", "plays", "game_plays", "drives", "ppd", "nh_neut", "pr_neut", "proe",
         "expl_off", "expl_def", "epa_off", "epa_def", "pts", "pts_allowed"]
MIN_TG = 64   # need >= 64 team-games (~2 weeks) of the current season before trusting its league mean


def rmse(p, a):
    return float(np.sqrt(np.nanmean((np.asarray(p, float) - np.asarray(a, float)) ** 2)))


def ols(y, X, cov="HC1", groups=None):
    X = sm.add_constant(X.astype(float), has_constant="add")
    if groups is not None:
        return sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    return sm.OLS(np.asarray(y, float), X).fit(cov_type=cov)


def fit_pred(tr, te, cols, y="total_pts", offset="lg_blend", groups=None):
    f = ols(tr[y] - tr[offset], tr[cols], groups=None if groups is None else tr[groups])
    return f, np.asarray(te[offset] + f.predict(sm.add_constant(te[cols].astype(float), has_constant="add")))


def load_games_table():
    m = pd.read_csv(HERE / "_game_features.csv", low_memory=False)
    m["dome"] = m.is_dome.astype(int)
    return m


def season_to_date_league_ref(feats=FEATS):
    """leak-free league reference per (season, gameday): mean of the raw team-game feature over all
    team-games of the same season that finished on strictly earlier dates. Returns a frame keyed by
    (season, gameday) with {f}_lgstd columns (NaN when < MIN_TG team-games so far)."""
    tg = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
    tg = tg[tg.game_type == "REG"].copy()
    tg["game_plays"] = tg.plays + tg.def_plays
    day = tg.groupby(["season", "gameday"])[feats].agg(["sum", "count"])
    day.columns = [f"{a}__{b}" for a, b in day.columns]
    day = day.reset_index().sort_values(["season", "gameday"])
    out = day[["season", "gameday"]].copy()
    for f in feats:
        cs = day.groupby("season")[f"{f}__sum"].cumsum() - day[f"{f}__sum"]
        cn = day.groupby("season")[f"{f}__count"].cumsum() - day[f"{f}__count"]
        out[f"{f}_lgstd"] = np.where(cn >= MIN_TG, cs / cn.replace(0, np.nan), np.nan)
    return out


def add_std_centered(m, feats=FEATS, v="r8"):
    """adds h_/a_{f}_{v}s = raw prior-8 feature minus the season-to-date league mean (fallback: the
    expert's own {f}_{v}d, i.e. prior-season reference) and the implied season-level offset."""
    ref = season_to_date_league_ref(feats)
    m = m.merge(ref, on=["season", "gameday"], how="left")
    new = {}
    for f in feats:
        for s in ("h", "a"):
            std = m[f"{s}_{f}_{v}"] - m[f"{f}_lgstd"]
            new[f"{s}_{f}_{v}s"] = std.fillna(m[f"{s}_{f}_{v}d"])
    return pd.concat([m, pd.DataFrame(new, index=m.index)], axis=1)


def combos(d, v):
    d = d.copy()
    d["spp_avg"] = (d[f"h_spp_neut_{v}"] + d[f"a_spp_neut_{v}"]) / 2
    d["sppr_avg"] = (d[f"h_spp_neut_run_{v}"] + d[f"a_spp_neut_run_{v}"]) / 2
    d["plays_sum"] = d[f"h_plays_{v}"] + d[f"a_plays_{v}"]
    d["gplays_avg"] = (d[f"h_game_plays_{v}"] + d[f"a_game_plays_{v}"]) / 2
    d["ppd_sum"] = d[f"h_ppd_{v}"] + d[f"a_ppd_{v}"]
    d["nh_sum"] = d[f"h_nh_neut_{v}"] + d[f"a_nh_neut_{v}"]
    d["pr_sum"] = d[f"h_pr_neut_{v}"] + d[f"a_pr_neut_{v}"]
    d["proe_sum"] = d[f"h_proe_{v}"] + d[f"a_proe_{v}"]
    d["expl_off_sum"] = d[f"h_expl_off_{v}"] + d[f"a_expl_off_{v}"]
    d["expl_def_sum"] = d[f"h_expl_def_{v}"] + d[f"a_expl_def_{v}"]
    d["epa_sum"] = d[f"h_epa_off_{v}"] + d[f"a_epa_off_{v}"] + d[f"h_epa_def_{v}"] + d[f"a_epa_def_{v}"]
    return d


def reg_sample(d, need):
    return d[(d.game_type == "REG") & d.mkt_total.notna() & d[need].notna().all(axis=1)].copy()


def ci_str(dm, lo, hi):
    return f"{dm:+.3f} [{lo:+.3f},{hi:+.3f}]"


def by_season(te, err_new, err_old, label):
    """per-season paired dMAE of a rule (err_new) vs a reference (err_old)."""
    rows = []
    for y, idx in te.groupby("season").indices.items():
        a = np.abs(np.asarray(err_new)[idx]); b = np.abs(np.asarray(err_old)[idx])
        rows.append((y, len(idx), a.mean() - b.mean()))
    return f"  {label:46s} per-season dMAE: " + "  ".join(f"{y}: {dm:+.3f} (n={n})" for y, n, dm in rows)
