"""00_sanity.py - sign conventions, coverage, baseline market error by season.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/00_sanity.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae

m = merged()
sp = m[m.mkt_spread.notna()]
print("games 2009-2025 with closing spread:", len(sp), "| seasons", sp.season.min(), "-", sp.season.max())
print("corr(mkt_spread, margin) =", round(np.corrcoef(sp.mkt_spread, sp.margin)[0, 1], 3), "(must be strongly negative)")
print("corr(spread_line, margin) =", round(np.corrcoef(sp.spread_line, sp.margin)[0, 1], 3), "(must be strongly positive)")
print("mean spread_err_mkt (margin + mkt_spread) =", round(sp.spread_err_mkt.mean(), 3),
      "| SD =", round(sp.spread_err_mkt.std(), 3), "| MAE =", round(mae(-sp.mkt_spread, sp.margin), 3))
print("mean total_err_mkt (total - line) =", round(sp.total_err_mkt.mean(), 3), "| SD =", round(sp.total_err_mkt.std(), 3))
# nfelo market columns: check they agree with nflverse closing line
j = sp[sp.home_line_close.notna()]
d = (j.home_line_close - j.mkt_spread)
print("nfelo home_line_close vs nflverse mkt_spread: n =", len(j), "| exact match share =", round((d.abs() < 0.01).mean(), 3),
      "| within 0.5 =", round((d.abs() <= 0.5).mean(), 3), "| mean diff =", round(d.mean(), 3))
jt = sp[sp.total_line_close.notna()]
dt = jt.total_line_close - jt.mkt_total
print("nfelo total_line_close vs nflverse total_line: n =", len(jt), "| exact =", round((dt.abs() < 0.01).mean(), 3), "| within 0.5 =", round((dt.abs() <= 0.5).mean(), 3))
jo = sp[sp.home_line_open.notna()]
print("home_line_open coverage by season:", sp.groupby("season").home_line_open.apply(lambda s: s.notna().mean()).round(2).to_dict())
print("total_line_open coverage by season:", sp.groupby("season").total_line_open.apply(lambda s: s.notna().mean()).round(2).to_dict())
print("\nBy season: n, market spread MAE, mean err, total MAE")
print(sp.groupby("season").apply(lambda d: pd.Series({"n": len(d), "sp_mae": mae(-d.mkt_spread, d.margin), "sp_bias": d.spread_err_mkt.mean(),
      "tot_mae": mae(d.mkt_total, d.total_pts), "tot_bias": d.total_err_mkt.mean()})).round(3).to_string())
