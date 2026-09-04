"""00: explore the schedule-related fields (rest, weekday, gametime, week) and nfelo's own
rest / time-zone modifiers so later scripts use the right definitions."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import merged, load_nfelo
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
m = merged()
print("rows", len(m), "REG", (m.game_type=="REG").sum(), "POST", (m.game_type!="REG").sum())
r = m[m.game_type=="REG"].copy()
print("\nhome_rest value counts (REG):"); print(r.home_rest.value_counts().sort_index().to_dict())
print("away_rest value counts (REG):"); print(r.away_rest.value_counts().sort_index().to_dict())
print("\nrest NA by season:", r.groupby("season").home_rest.apply(lambda s: s.isna().sum()).to_dict())
print("\nweek 1 rest values:", r[r.week==1].home_rest.value_counts().to_dict())
print("\nweekday counts:", r.weekday.value_counts().to_dict())
print("gametime counts (top 15):", r.gametime.value_counts().head(15).to_dict())
print("gametime NA:", r.gametime.isna().sum())
print("\nrest by weekday (home):"); print(pd.crosstab(r.weekday, r.home_rest.clip(upper=15)))
print("\nnfelo home_bye_mod values:", m.home_bye_mod.value_counts().head(8).to_dict())
print("nfelo away_bye_mod values:", m.away_bye_mod.value_counts().head(8).to_dict())
print("nfelo home_time_advantage_mod values:", m.home_time_advantage_mod.round(2).value_counts().head(12).to_dict())
print("\ncrosstab home_rest>=13 vs nfelo home_bye_mod>0 (2009-2025 REG):")
print(pd.crosstab(r.home_rest>=13, r.home_bye_mod.fillna(0)>0))
print("crosstab away_rest>=13 vs nfelo away_bye_mod<0? (values):", r.away_bye_mod.dropna().unique()[:10])
print(pd.crosstab(r.away_rest>=13, r.away_bye_mod.fillna(0)!=0))
# what bye mods look like by rest value
print("\nmean home_bye_mod by home_rest:"); print(r.groupby(r.home_rest.clip(upper=16)).home_bye_mod.agg(["mean","count"]).round(2))
print("\nmean away_bye_mod by away_rest:"); print(r.groupby(r.away_rest.clip(upper=16)).away_bye_mod.agg(["mean","count"]).round(2))
# nfelo line conversion: how does nfelo_home_line_close relate to nfelo_dif?
n = load_nfelo().dropna(subset=["nfelo_dif_close","nfelo_home_line_close"])
b = np.polyfit(n.nfelo_dif_close, n.nfelo_home_line_close, 1)
print("\nnfelo_home_line_close = %.4f*nfelo_dif_close + %.3f  (1/25=%.4f)" % (b[0], b[1], 1/25))
b2 = np.polyfit(n.nfelo_dif_base, n.nfelo_home_line_close, 1)
print("nfelo_home_line_close = %.4f*nfelo_dif_base + %.3f ; corr=%.3f" % (b2[0], b2[1], np.corrcoef(n.nfelo_dif_base, n.nfelo_home_line_close)[0,1]))
print("nfelo_dif_close - nfelo_dif_base describe:"); print((n.nfelo_dif_close - n.nfelo_dif_base).describe().round(2))
# historic projected spreads coverage
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
print("\nhistoric_projected_spreads seasons:", h.season.min(), h.season.max(), "rows", len(h))
print("cols sample:", h[["season","week","home_line_close","home_line_pre_regression","home_dif","home_dif_pre_reg","market_regression_factor","home_net_bye_mod","time_mod"]].head())
print("market_regression_factor describe:", h.market_regression_factor.describe().round(3).to_dict())
# season-level: sanity check convention
print("\nsign check corr(mkt_spread, margin):", round(np.corrcoef(r.dropna(subset=["mkt_spread"]).mkt_spread, r.dropna(subset=["mkt_spread"]).margin)[0,1],3))
