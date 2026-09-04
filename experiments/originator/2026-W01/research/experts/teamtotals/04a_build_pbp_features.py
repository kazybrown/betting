"""Build per team-game offense/defense profile stats from play-by-play:
  nflscrapR 2009-2019 (regular season) and nflfastR 2023-2025 (REG + POST).
Outputs research/experts/teamtotals/_pbp_teamgame.csv (intermediate data for 05_*).

Per team-game (offense = posteam side; defense = defteam side, i.e. what the team allowed):
  plays, expl20 (scrimmage plays >= 20 yds), expl15, epa_sum, succ (epa>0), pass_plays,
  rz_trips (drives reaching yardline_100 <= 20), rz_td (those ending in a posteam TD),
  pace_secs / pace_plays (seconds between consecutive scrimmage plays in the same drive,
  neutral situations: |score diff| <= 8 and outside the final quarter).
"""
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kit import norm

OUT = Path(__file__).resolve().parent / "_pbp_teamgame.csv"
SCRAPR = "/home/user/nflscrapR-data/play_by_play_data/regular_season/reg_pbp_{}.csv"
FASTR = "/home/user/NFLkz/data/cache/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "home_team", "away_team", "posteam", "defteam", "play_type", "yards_gained", "yardline_100",
        "touchdown", "td_team", "drive", "game_seconds_remaining", "epa", "score_differential", "play_id"]


def team_game_stats(d, key_col, season, drive_col="drive"):
    d = d.copy()
    d["posteam"] = d.posteam.map(lambda t: norm(t) if isinstance(t, str) else t)
    d["defteam"] = d.defteam.map(lambda t: norm(t) if isinstance(t, str) else t)
    d["td_team"] = d.td_team.map(lambda t: norm(t) if isinstance(t, str) else t)
    d = d[d.posteam.notna() & d.defteam.notna()]
    scrim = d[d.play_type.isin(["pass", "run"])].copy()
    scrim["expl20"] = (scrim.yards_gained >= 20).astype(int)
    scrim["expl15"] = (scrim.yards_gained >= 15).astype(int)
    scrim["succ"] = (scrim.epa > 0).astype(int)
    scrim["is_pass"] = (scrim.play_type == "pass").astype(int)
    scrim = scrim.sort_values([key_col, "play_id"])
    # pace: seconds between consecutive scrimmage plays within same game/drive/posteam, neutral situations
    grp = scrim.groupby([key_col, drive_col, "posteam"])
    dt = grp.game_seconds_remaining.shift(1) - scrim.game_seconds_remaining
    neutral = (scrim.score_differential.abs() <= 8) & (scrim.game_seconds_remaining > 900)
    ok = dt.notna() & (dt > 0) & (dt <= 60) & neutral
    scrim["pace_secs"] = np.where(ok, dt, 0.0)
    scrim["pace_plays"] = ok.astype(int)
    agg = scrim.groupby([key_col, "posteam", "defteam"]).agg(
        plays=("play_id", "size"), expl20=("expl20", "sum"), expl15=("expl15", "sum"), epa_sum=("epa", "sum"),
        succ=("succ", "sum"), pass_plays=("is_pass", "sum"), pace_secs=("pace_secs", "sum"), pace_plays=("pace_plays", "sum")).reset_index()
    # red zone by drive (all plays with a posteam, incl. penalties)
    dd = d[d.yardline_100.notna()].groupby([key_col, drive_col, "posteam"]).agg(
        min_yl=("yardline_100", "min"), td=("touchdown", "max"), td_team=("td_team", "last")).reset_index()
    dd["rz"] = (dd.min_yl <= 20).astype(int)
    dd["rz_td"] = ((dd.min_yl <= 20) & (dd.td == 1) & (dd.td_team == dd.posteam)).astype(int)
    rz = dd.groupby([key_col, "posteam"]).agg(rz_trips=("rz", "sum"), rz_td=("rz_td", "sum")).reset_index()
    agg = agg.merge(rz, on=[key_col, "posteam"], how="left")
    agg["season"] = season
    return agg


frames = []
for yr in range(2009, 2020):
    d = pd.read_csv(SCRAPR.format(yr), usecols=COLS + ["game_date"], low_memory=False)
    a = team_game_stats(d, "game_id", yr)
    dates = d.groupby("game_id").game_date.first()
    a["game_date"] = a.game_id.map(dates)
    a["old_game_id"] = a.game_id.astype(np.int64)
    a["game_id"] = ""
    frames.append(a)
    print(yr, len(a), "team-games", flush=True)
for yr in (2023, 2024, 2025):
    d = pd.read_parquet(FASTR.format(yr), columns=COLS + ["fixed_drive", "game_date", "season_type"])
    d["drive"] = d.fixed_drive
    a = team_game_stats(d, "game_id", yr, drive_col="drive")
    dates = d.groupby("game_id").game_date.first()
    a["game_date"] = a.game_id.map(dates)
    a["old_game_id"] = -1
    frames.append(a)
    print(yr, len(a), "team-games", flush=True)
tg = pd.concat(frames, ignore_index=True)
tg = tg.rename(columns={"posteam": "team", "defteam": "opp"})
tg.to_csv(OUT, index=False)
print("wrote", OUT, tg.shape)
print(tg.groupby("season").agg(tg=("team", "size"), plays=("plays", "mean"), expl20=("expl20", "mean"), rz=("rz_trips", "mean"),
      rz_td=("rz_td", "mean"), pace=("pace_secs", "sum")).assign(pace=lambda x: x.pace / tg.groupby("season").pace_plays.sum()).round(2).to_string())
