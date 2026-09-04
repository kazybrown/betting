"""Shared feature builder for the totals expert. Everything is LEAK-FREE: every team /
league feature attached to a game uses only games that finished before it.

build(K_team=6, K_lg=64) returns one row per scored game (1999-2025, REG+POST) with:
  total_pts, mkt_total (closing), total_err_mkt = total_pts - mkt_total
  Elo (2009+ via nfelo): home_pts_vs_avg, away_pts_vs_avg, elo_sum, elo_dif_pts, hfa_pts
  Team scoring proxies (prefix h_/a_ = home/away team):
     pf, pa      blended points-for / points-against per game:
                 (K_team * prior-season mean + gp * season-to-date mean) / (K_team + gp)
     gt          same blend for the team's game totals (pf+pa)
     pf_prev, pa_prev, gt_prev   prior-season REG means (league mean if no history)
     pf_ytd, pa_ytd, gp          season-to-date (before this game) means / games played
  League prior: lg_prev (prior-season REG mean total), lg_ytd (mean total of all games
     already played this season), lg_blend = (K_lg*lg_prev + n*lg_ytd)/(K_lg+n)
  Environment: is_dome (dome/closed), roof, div_game, neutral, week, temp, wind (outdoor
     only), h_rest/a_rest, short-week / bye flags, grass
  Pace (2009-2019, 2023-2025 only; NaN elsewhere), same blend as pf/pa:
     h_/a_ plays, sec_per_play, no_huddle_rate, pass_rate, off_epa, def_epa
  Precipitation flags (2023-2025 only): precip_any, precip_strict
Conventions per README: margin = home - away; mkt_spread negative = home favored.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from kit import load_games, load_nfelo, norm  # noqa: E402

TRAIN_MAX = 2021
TEST_SEASONS = (2022, 2023, 2024, 2025)
PACE_COLS = ["plays", "sec_per_play", "no_huddle_rate", "pass_rate", "off_epa", "def_epa"]


def _long(g):
    """two rows per game: (team, opp, pf, pa, home flag)"""
    h = g[["gid", "season", "week", "gameday", "game_type", "home", "away", "home_score", "away_score"]].copy()
    h.columns = ["gid", "season", "week", "gameday", "game_type", "team", "opp", "pf", "pa"]; h["is_home"] = 1
    a = g[["gid", "season", "week", "gameday", "game_type", "away", "home", "away_score", "home_score"]].copy()
    a.columns = ["gid", "season", "week", "gameday", "game_type", "team", "opp", "pf", "pa"]; a["is_home"] = 0
    return pd.concat([h, a], ignore_index=True)


def _blend_team_stats(tg, cols, K, prefix=""):
    """tg: team-game long frame sorted by (team, season, gameday, gid). For each col add
    {col}_ytd (season-to-date mean BEFORE this game), {col}_prev (prior-season REG mean),
    {col} (blend). League mean of the prior season fills missing prev."""
    tg = tg.sort_values(["team", "season", "gameday", "gid"]).copy()
    grp = tg.groupby(["team", "season"])
    tg["gp"] = grp.cumcount()
    for c in cols:
        tg[f"{c}_ytd"] = grp[c].transform(lambda s: s.shift(1).expanding().mean())
    prev = tg[tg.game_type == "REG"].groupby(["team", "season"])[cols].mean().reset_index()
    prev["season"] = prev.season + 1
    prev = prev.rename(columns={c: f"{c}_prev" for c in cols})
    lg = tg[tg.game_type == "REG"].groupby("season")[cols].mean().reset_index()
    lg["season"] = lg.season + 1
    lg = lg.rename(columns={c: f"{c}_lgprev" for c in cols})
    tg = tg.merge(prev, on=["team", "season"], how="left").merge(lg, on="season", how="left")
    # leak-free league season-to-date mean (games on strictly earlier dates this season):
    # last-resort prior when neither the team nor the league has a prior-season value
    tg = tg.sort_values(["team", "season", "gameday", "gid"]).copy()
    day = tg.groupby(["season", "gameday"])[cols].agg(["sum", "count"])
    day.columns = [f"{a}_{b}" for a, b in day.columns]
    day = day.reset_index().sort_values(["season", "gameday"])
    for c in cols:
        cs = day.groupby("season")[f"{c}_sum"].cumsum() - day[f"{c}_sum"]
        cn = day.groupby("season")[f"{c}_count"].cumsum() - day[f"{c}_count"]
        day[f"{c}_lgytd"] = np.where(cn > 0, cs / cn.replace(0, np.nan), np.nan)
    tg = tg.merge(day[["season", "gameday"] + [f"{c}_lgytd" for c in cols]], on=["season", "gameday"], how="left")
    for c in cols:
        tg[f"{c}_prev"] = tg[f"{c}_prev"].fillna(tg[f"{c}_lgprev"]).fillna(tg[f"{c}_lgytd"])
        prior = tg[f"{c}_prev"]
        ytd = tg[f"{c}_ytd"].fillna(prior)
        tg[c + "_blend"] = (K * prior + tg.gp * ytd) / (K + tg.gp)
    return tg


def build(K_team=6, K_lg=64, min_season=1999, verbose=True):
    g = load_games(min_season=min_season)
    g = g.sort_values(["gameday", "gid"]).reset_index(drop=True)
    # ---------------- team scoring proxies ----------------
    tg = _long(g)
    tg["gt"] = tg.pf + tg.pa
    tg = _blend_team_stats(tg, ["pf", "pa", "gt"], K_team)
    keep = ["gid", "team", "gp"] + [f"{c}{s}" for c in ["pf", "pa", "gt"] for s in ["_ytd", "_prev", "_blend"]]
    tg = tg[keep].rename(columns={f"{c}_blend": c for c in ["pf", "pa", "gt"]})
    # ---------------- pace (pbp) ----------------
    pf_path = HERE / "pace_team_games.csv"
    if pf_path.exists():
        pc = pd.read_csv(pf_path)
        pc["team"] = pc.team.map(norm)
        base = _long(g)[["gid", "season", "week", "gameday", "game_type", "team"]]
        pc = base.merge(pc.drop(columns=["season"]), on=["gid", "team"], how="left")
        # only seasons where pbp exists count toward ytd / prev; others stay NaN
        has = pc.groupby("season").plays.transform(lambda s: s.notna().mean()) > 0.5
        pc = pc[has].copy()
        pc = _blend_team_stats(pc, PACE_COLS, K_team)
        pc = pc[["gid", "team"] + [c + "_blend" for c in PACE_COLS]].rename(columns={c + "_blend": c for c in PACE_COLS})
        tg = tg.merge(pc, on=["gid", "team"], how="left")
    else:
        for c in PACE_COLS:
            tg[c] = np.nan
    # attach to games
    hcols = {c: "h_" + c for c in tg.columns if c not in ("gid", "team")}
    acols = {c: "a_" + c for c in tg.columns if c not in ("gid", "team")}
    m = g.merge(tg.rename(columns=hcols).rename(columns={"team": "home"}), on=["gid", "home"], how="left")
    m = m.merge(tg.rename(columns=acols).rename(columns={"team": "away"}), on=["gid", "away"], how="left")
    # ---------------- league prior ----------------
    reg = m[m.game_type == "REG"]
    lg_prev = reg.groupby("season").total_pts.mean().rename("lg_prev").reset_index()
    lg_prev["season"] = lg_prev.season + 1
    m = m.merge(lg_prev, on="season", how="left")
    m = m.sort_values(["gameday", "gid"]).reset_index(drop=True)
    # games played on strictly earlier dates in the same season
    day = m.groupby(["season", "gameday"]).agg(n=("total_pts", "size"), s=("total_pts", "sum")).reset_index()
    day = day.sort_values(["season", "gameday"])
    day["n_before"] = day.groupby("season").n.cumsum() - day.n
    day["s_before"] = day.groupby("season").s.cumsum() - day.s
    m = m.merge(day[["season", "gameday", "n_before", "s_before"]], on=["season", "gameday"], how="left")
    m["lg_ytd"] = np.where(m.n_before > 0, m.s_before / m.n_before.replace(0, np.nan), np.nan)
    m["lg_blend"] = (K_lg * m.lg_prev + m.n_before * m.lg_ytd.fillna(m.lg_prev)) / (K_lg + m.n_before)
    # ---------------- nfelo ----------------
    n = load_nfelo()
    keep = ["gid", "starting_nfelo_home", "starting_nfelo_away", "elo_dif_pts", "home_pts_vs_avg", "away_pts_vs_avg",
            "hfa_pts", "home_538_qb_adj", "away_538_qb_adj", "nfelo_home_line_close", "home_line_close", "total_line_close"]
    m = m.merge(n[keep], on="gid", how="left")
    m["elo_sum"] = m.home_pts_vs_avg + m.away_pts_vs_avg
    m["qb_sum"] = (m.home_538_qb_adj.fillna(0) + m.away_538_qb_adj.fillna(0)) / 25.0
    # ---------------- environment ----------------
    m["h_rest"] = m.home_rest; m["a_rest"] = m.away_rest
    m["short_home"] = (m.home_rest <= 5).astype(int); m["short_away"] = (m.away_rest <= 5).astype(int)
    m["bye_home"] = (m.home_rest >= 13).astype(int); m["bye_away"] = (m.away_rest >= 13).astype(int)
    m["grass"] = m.surface.fillna("").str.strip().eq("grass").astype(int)
    m["div"] = m.div_game.fillna(0).astype(int)
    m["post"] = (m.game_type != "REG").astype(int)
    m["outdoor"] = m.roof.isin(["outdoors", "open"]).astype(int)
    pp = HERE / "precip_games.csv"
    if pp.exists():
        p = pd.read_csv(pp)
        # nflfastR weather string (2023-25) fills nflverse temp/wind gaps (agreement r=0.97/0.999 where both exist)
        p["temp_s"] = p.weather.str.extract(r"Temp:\s*(-?\d+)")[0].astype(float)
        p["wind_s"] = p.weather.str.extract(r"Wind:.*?(\d+)\s*mph")[0].astype(float)
        m = m.merge(p[["gid", "precip_any", "precip_strict", "temp_s", "wind_s"]], on="gid", how="left")
        fill = (m.outdoor == 1)
        m.loc[fill, "temp"] = m.loc[fill, "temp"].fillna(m.loc[fill, "temp_s"])
        m.loc[fill, "wind"] = m.loc[fill, "wind"].fillna(m.loc[fill, "wind_s"])
    else:
        m["precip_any"] = np.nan; m["precip_strict"] = np.nan
    # model features: 0 wind / 70F indoors; outdoor games with unknown weather (mostly 2022,
    # 46% of outdoor games) get a league-typical value + a flag. Raw temp/wind stay NaN there.
    m["wx_missing"] = ((m.outdoor == 1) & m.wind.isna()).astype(int)
    typ_t = m[m.outdoor == 1].groupby("week").temp.transform("median")
    m["wind_f"] = np.where(m.outdoor == 1, m.wind.fillna(8.0), 0.0)
    m["temp_f"] = np.where(m.outdoor == 1, m.temp.fillna(typ_t).fillna(58.0), 70.0)
    # convenience combos
    m["pf_sum"] = m.h_pf + m.a_pf; m["pa_sum"] = m.h_pa + m.a_pa
    m["gt_avg"] = (m.h_gt + m.a_gt) / 2
    m["h_off_vs_a_def"] = m.h_pf + m.a_pa; m["a_off_vs_h_def"] = m.a_pf + m.h_pa
    m["train"] = m.season <= TRAIN_MAX; m["test"] = m.season.isin(TEST_SEASONS)
    if verbose:
        r = m[m.mkt_total.notna()]
        print(f"[common] games={len(m)} seasons {m.season.min()}-{m.season.max()} | with market total={len(r)} "
              f"| nfelo elo coverage 2009+ = {m.loc[m.season>=2009,'elo_sum'].notna().mean():.3f} "
              f"| pace coverage = {m.h_plays.notna().mean():.3f}")
        print(f"[common] sign check corr(mkt_spread, margin) = {np.corrcoef(r.mkt_spread, r.margin)[0,1]:.3f} (must be strongly NEGATIVE); "
              f"corr(mkt_total, total_pts) = {np.corrcoef(r.mkt_total, r.total_pts)[0,1]:.3f} (positive)")
    return m


def mae(pred, actual):
    return float(np.nanmean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def paired_mae_ci(err_a, err_b, n_boot=2000, seed=0):
    """bootstrap CI for MAE(a) - MAE(b) on the same games (negative = a better)."""
    rng = np.random.default_rng(seed)
    a = np.abs(np.asarray(err_a, float)); b = np.abs(np.asarray(err_b, float))
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    d = a - b
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(d))


def ou_rate(pred_total, mkt_total, actual):
    """over/under hit rate of taking OUR side vs the market total (pred>mkt -> over)."""
    p, mk, y = map(lambda x: np.asarray(x, float), (pred_total, mkt_total, actual))
    ok = ~(np.isnan(p) | np.isnan(mk) | np.isnan(y)); p, mk, y = p[ok], mk[ok], y[ok]
    pick_over = p > mk; push = y == mk; over = y > mk
    w = int(np.sum((pick_over & over & ~push) | (~pick_over & ~over & ~push)))
    l = int(np.sum(~push)) - w
    return w, l, int(push.sum())


if __name__ == "__main__":
    m = build()
    print(m[["season", "gid", "total_pts", "mkt_total", "lg_prev", "lg_blend", "h_pf", "h_pa", "a_pf", "a_pa", "elo_sum", "h_plays", "h_sec_per_play"]].tail(5).to_string())
    print(m[["lg_prev", "lg_ytd", "lg_blend", "h_pf", "h_pa", "h_gt", "elo_sum", "h_plays", "h_sec_per_play", "wind_f", "temp_f"]].describe().round(2).to_string())
