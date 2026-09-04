"""Shared loaders for the ORIGINATOR research kit (pandas)."""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "STL": "LA", "SD": "LAC", "JAC": "JAX", "WSH": "WAS"}


def norm(t):
    return TEAM_FIX.get(t, t)


def load_games(min_season=2009, scored_only=True):
    """nflverse games with derived fields. Conventions: margin = home - away;
    mkt_spread is in ORIGINATOR convention (negative = home favored) = -spread_line."""
    g = pd.read_csv(D / "games_1999_2025.csv", low_memory=False)
    g = g[g.season >= min_season].copy()
    if scored_only:
        g = g[g.result.notna()].copy()
    g["home"] = g.home_team.map(norm)
    g["away"] = g.away_team.map(norm)
    g["margin"] = g.home_score - g.away_score
    g["total_pts"] = g.home_score + g.away_score
    g["mkt_spread"] = -g.spread_line          # ORIGINATOR convention
    g["mkt_total"] = g.total_line
    g["spread_err_mkt"] = g.margin + g.mkt_spread   # 0 = market exactly right
    g["total_err_mkt"] = g.total_pts - g.mkt_total
    g["is_dome"] = g.roof.isin(["dome", "closed"])
    g["neutral"] = g.location.eq("Neutral")
    g["gid"] = g.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
    return g


def load_nfelo():
    n = pd.read_csv(D / "nfelo_games.csv", low_memory=False)
    n = n.rename(columns={n.columns[0]: "row"})
    parts = n.game_id.str.split("_", expand=True)
    n["season"] = parts[0].astype(int)
    n["week"] = parts[1].astype(int)
    n["away"] = parts[2].map(norm)
    n["home"] = parts[3].map(norm)
    n["gid"] = n.season.astype(str) + "_" + parts[1] + "_" + n.away + "_" + n.home
    n["elo_dif_pts"] = (n.starting_nfelo_home - n.starting_nfelo_away) / 25.0
    n["home_pts_vs_avg"] = (n.starting_nfelo_home - 1505) / 25.0
    n["away_pts_vs_avg"] = (n.starting_nfelo_away - 1505) / 25.0
    n["hfa_pts"] = (n.hfa_mod + n.home_time_advantage_mod.fillna(0)) / 25.0
    return n


def merged(min_season=2009):
    """games joined to nfelo on normalized game id (regular + post season)."""
    g = load_games(min_season)
    n = load_nfelo()
    keep = ["gid", "starting_nfelo_home", "starting_nfelo_away", "elo_dif_pts",
            "home_pts_vs_avg", "away_pts_vs_avg", "hfa_pts", "hfa_mod",
            "home_538_qb_adj", "away_538_qb_adj", "home_net_qb_mod", "div_game_mod",
            "dif_surface_mod", "home_time_advantage_mod", "home_bye_mod", "away_bye_mod",
            "nfelo_home_line_open", "nfelo_home_line_close", "nfelo_home_probability_close",
            "home_line_open", "home_line_close", "total_line_open", "total_line_close"]
    keep = [k for k in keep if k in n.columns]
    m = g.merge(n[keep], on="gid", how="left")
    return m


def mae(pred, actual):
    return float(np.nanmean(np.abs(np.asarray(pred) - np.asarray(actual))))


def ats(pred_spread, mkt_spread, margin):
    """ATS record of taking the side our number favors vs the market number.
    pred/mkt in ORIGINATOR convention; margin = home - away. Returns (wins, losses, pushes)."""
    pred, mkt, mg = map(np.asarray, (pred_spread, mkt_spread, margin))
    pick_home = pred < mkt                      # we like home more than the market does
    cover_home = (mg + mkt) > 0
    push = (mg + mkt) == 0
    w = np.sum((pick_home & cover_home & ~push) | (~pick_home & ~cover_home & ~push))
    l = np.sum(~push) - w
    return int(w), int(l), int(np.sum(push))


if __name__ == "__main__":
    m = merged()
    print("merged rows:", len(m), "| seasons", m.season.min(), "-", m.season.max())
    cov = m.groupby("season").nfelo_home_line_close.apply(lambda s: s.notna().mean()).round(2)
    print("nfelo join coverage by season:", cov.to_dict())
    reg = m[(m.game_type == "REG") & m.nfelo_home_line_close.notna() & m.mkt_spread.notna()]
    print("market close MAE (spread):", round(mae(-reg.mkt_spread, reg.margin), 3),
          "| nfelo close MAE:", round(mae(-reg.nfelo_home_line_close, reg.margin), 3),
          "| market total MAE:", round(mae(reg.mkt_total, reg.total_pts), 3))
    print("sign check: corr(mkt_spread, margin) =", round(np.corrcoef(reg.mkt_spread, reg.margin)[0, 1], 3), "(should be strongly NEGATIVE)")
    w, l, p = ats(reg.nfelo_home_line_close, reg.mkt_spread, reg.margin)
    print(f"nfelo close vs market close ATS: {w}-{l}-{p} ({w/(w+l):.3f})")
