"""Exploration: field distributions relevant to HFA theories."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import load_games, load_nfelo, merged
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

g = load_games()
n = load_nfelo()
m = merged()
print("games rows", len(g), "| game_type", g.game_type.value_counts().to_dict())
print("location", g.location.value_counts().to_dict())
print("weekday", g.weekday.value_counts().to_dict())
print("roof", g.roof.value_counts(dropna=False).to_dict())
print("gametime sample", g.gametime.dropna().unique()[:20])
print("gametime null by season", g.groupby("season").gametime.apply(lambda s: s.isna().mean()).round(2).to_dict())
print("\nNeutral games by season/stadium:")
print(g[g.neutral].groupby(["season"]).size().to_dict())
print(g[g.neutral][["season","week","game_type","away","home","stadium","result","spread_line"]].to_string())
print("\nInternational-looking stadiums (non-neutral too):")
intl = g[g.stadium.str.contains("Wembley|Tottenham|Twickenham|Azteca|Allianz|Deutsche|Frankfurt|Arena Corinthians|Neo Quimica|Accor|Bernabeu|Croke|Olympiastadion|Tottenham", case=False, na=False)]
print(intl.groupby(["season","stadium","location"]).size())
print("\nnfelo hfa_mod describe:"); print(n.hfa_mod.describe())
print("hfa_base_mod describe:"); print(n.hfa_base_mod.describe())
print("home_time_advantage_mod:"); print(n.home_time_advantage_mod.describe())
print("hfa_pts by season mean:"); print(n.groupby("season").hfa_pts.agg(["mean","std","min","max"]).round(2))
print("\nhfa_mod unique count", n.hfa_mod.nunique())
print("hfa_pts by home team (2022-2025 mean):")
print(n[n.season>=2022].groupby("home").hfa_pts.mean().round(2).sort_values().to_string())
# check whether nfelo hfa_mod is 0 in neutral games
mm = m[m.neutral]
print("\nneutral games nfelo hfa_pts:", mm[["season","week","away","home","hfa_pts","hfa_mod","home_time_advantage_mod"]].to_string())
# distribution of home_rest/away_rest
print("\nrest days", g.home_rest.value_counts().head(10).to_dict())
# div_game col
print("div_game", g.div_game.value_counts().to_dict())
# check how nfelo line relates to elo diff + hfa
r = m.dropna(subset=["nfelo_home_line_close"])
print("\ncheck: nfelo_dif_base vs elo_dif + hfa? corr", np.corrcoef(-(r.elo_dif_pts + r.hfa_pts), r.nfelo_home_line_close)[0,1])
print("gametime by weekday top", g.groupby("weekday").gametime.agg(lambda s: s.value_counts().head(4).to_dict()))
