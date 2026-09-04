"""00: Aggregate play-by-play to TEAM-GAME pace / efficiency metrics (and a game-level
precipitation flag for 2023-2025 from the nflfastR weather string).

Sources (README): nflscrapR reg-season pbp 2009-2019 (join on games.old_game_id) and
nflfastR pbp 2023-2025 (join on games.game_id). 2020-2022 pbp is NOT available locally.

Output: experts/totals/pace_team_games.csv, one row per (gid, team) with
  plays          offensive pass+run plays
  drives         offensive drives
  sec_per_play   possession seconds per offensive play (drive clock span / plays)
  no_huddle_rate share of pass+run plays flagged no_huddle
  pass_rate      pass share of pass+run plays
  off_epa, def_epa   EPA per pass/run play when the team has / does not have the ball
and experts/totals/precip_games.csv (gid, weather, precip_any, precip_strict) for 2023-25.
Re-runnable; prints coverage counts.
"""
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm

OUT = "/home/user/originator-2026-w01/research/experts/totals/"
g = load_games(min_season=2009)
old2gid = dict(zip(g.old_game_id.astype(str), g.gid))
new2gid = dict(zip(g.game_id, g.gid))


def agg_team_games(df, drive_col):
    """df: pbp rows with gid, posteam, defteam, play_type, gsr, drive, no_huddle, epa."""
    df = df[df.posteam.notna()].copy()
    df["posteam"] = df.posteam.map(norm); df["defteam"] = df.defteam.map(norm)
    pr = df[df.play_type.isin(["pass", "run"])].copy()
    pr["is_pass"] = (pr.play_type == "pass").astype(float)
    off = pr.groupby(["gid", "posteam"]).agg(plays=("play_type", "size"), no_huddle_rate=("no_huddle", "mean"),
                                             pass_rate=("is_pass", "mean"), off_epa=("epa", "mean")).reset_index()
    de = pr.groupby(["gid", "defteam"]).agg(def_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "posteam"})
    # possession clock: per drive, span of game_seconds_remaining over all rows with a posteam
    dr = df.dropna(subset=[drive_col, "game_seconds_remaining"]).groupby(["gid", "posteam", drive_col]).game_seconds_remaining.agg(["max", "min"])
    dr["span"] = dr["max"] - dr["min"]
    poss = dr.groupby(level=[0, 1]).agg(poss_sec=("span", "sum"), drives=("span", "size")).reset_index()
    t = off.merge(de, on=["gid", "posteam"], how="left").merge(poss, on=["gid", "posteam"], how="left")
    t["sec_per_play"] = t.poss_sec / t.plays
    return t.rename(columns={"posteam": "team"})


parts = []
for y in range(2009, 2020):
    df = pd.read_csv(f"/home/user/nflscrapR-data/play_by_play_data/regular_season/reg_pbp_{y}.csv",
                     usecols=["game_id", "posteam", "defteam", "play_type", "game_seconds_remaining", "drive", "no_huddle", "epa"],
                     low_memory=False)
    df["gid"] = df.game_id.astype(str).map(old2gid)
    miss = df.gid.isna().mean()
    t = agg_team_games(df, "drive"); t["season"] = y; parts.append(t)
    print(f"nflscrapR {y}: games={t.gid.nunique()} team-games={len(t)} unmapped rows={miss:.3f}")

precip = []
for y in (2023, 2024, 2025):
    df = pd.read_parquet(f"/home/user/NFLkz/data/cache/pbp/play_by_play_{y}.parquet",
                         columns=["game_id", "posteam", "defteam", "play_type", "game_seconds_remaining", "fixed_drive", "no_huddle", "epa", "weather"])
    df["gid"] = df.game_id.map(new2gid)
    t = agg_team_games(df, "fixed_drive"); t["season"] = y; parts.append(t)
    w = df.groupby("gid").weather.first().reset_index()
    w["w"] = w.weather.fillna("").str.lower()
    kw = r"rain|snow|shower|drizzle|sleet|flurr|storm|wet"
    w["precip_any"] = w.w.str.contains(kw).astype(int)
    hedge = r"chance|threat|possible|may|%\s*chance|clearing|earlier|before"
    w["precip_strict"] = (w.precip_any.eq(1) & ~w.w.str.contains(hedge)).astype(int)
    precip.append(w[["gid", "weather", "precip_any", "precip_strict"]])
    print(f"nflfastR {y}: games={t.gid.nunique()} team-games={len(t)} precip_any={int(w.precip_any.sum())} precip_strict={int(w.precip_strict.sum())}")

pace = pd.concat(parts, ignore_index=True)
pace.to_csv(OUT + "pace_team_games.csv", index=False)
pd.concat(precip, ignore_index=True).to_csv(OUT + "precip_games.csv", index=False)
print("wrote", len(pace), "team-games; summary:")
print(pace[["plays", "drives", "sec_per_play", "no_huddle_rate", "pass_rate", "off_epa", "def_epa"]].describe().round(3).to_string())
