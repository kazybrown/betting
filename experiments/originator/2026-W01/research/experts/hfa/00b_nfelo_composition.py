"""Check how nfelo_dif_base decomposes so rating-implied margin uses the right pieces."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import load_nfelo
n = load_nfelo()
n = n.dropna(subset=["nfelo_dif_base"])
comp = (n.starting_nfelo_home - n.starting_nfelo_away + n.hfa_mod + n.home_net_qb_mod.fillna(0)
        + n.home_net_bye_mod.fillna(0) + n.div_game_mod.fillna(0) + n.dif_surface_mod.fillna(0)
        + n.home_time_advantage_mod.fillna(0))
print("resid (full sum incl time adv) mean/std:", (n.nfelo_dif_base - comp).mean(), (n.nfelo_dif_base - comp).std())
comp2 = comp - n.home_time_advantage_mod.fillna(0)
print("resid (excluding time adv) mean/std:", (n.nfelo_dif_base - comp2).mean(), (n.nfelo_dif_base - comp2).std())
comp3 = comp2 - n.home_net_bye_mod.fillna(0) + n.home_bye_mod.fillna(0) - n.away_bye_mod.fillna(0)
print("resid (bye as home-away) mean/std:", (n.nfelo_dif_base - comp3).mean(), (n.nfelo_dif_base - comp3).std())
print("does hfa_mod already include time adv? corr(hfa_mod - hfa_base_mod, time_adv):",
      np.corrcoef((n.hfa_mod - n.hfa_base_mod), n.home_time_advantage_mod.fillna(0))[0,1])
print("mean hfa_mod - hfa_base_mod - time_adv:", (n.hfa_mod - n.hfa_base_mod - n.home_time_advantage_mod.fillna(0)).describe())
print("nfelo_dif_close vs nfelo_home_line_close: ", np.corrcoef(n.nfelo_dif_close, n.nfelo_home_line_close)[0,1])
print(n[["nfelo_dif_base","nfelo_dif_close","nfelo_home_line_close","home_line_close"]].head())
# how is dif converted to points? regress line on dif
b = np.polyfit(n.nfelo_dif_close, n.nfelo_home_line_close, 1); print("line = %.4f*dif + %.3f" % tuple(b))
print("home_net_qb_mod describe", n.home_net_qb_mod.describe())
print("div_game_mod values", n.div_game_mod.value_counts().head())
print("dif_surface_mod values", n.dif_surface_mod.value_counts().head())
print("home_bye_mod values", n.home_bye_mod.value_counts().head(), n.away_bye_mod.value_counts().head())
