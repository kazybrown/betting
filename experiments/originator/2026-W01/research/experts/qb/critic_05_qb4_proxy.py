"""CRITIC 05 (theory qb-4): the expert's runtime rule needs A_nfelo = points nfelo already charges for the backup, proposed as
-(team's 538 qb adj now - its adj in the last game with the starter)/25.  Check that proxy at stint-1 / stint-2-3 / stint-4+
events against the regression-based embedded value (2.3 / 1.6 / 0.8 pts) and look at its dispersion.  Also: what fraction of
the market-implied penalty does the proxy explain game-by-game (does a per-game A_nfelo beat the flat 2.3)?  Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna()].copy()
m["late"] = ((m.game_type != "REG") | (m.week >= 17)).astype(int)
# team-game table with own 538 adj (pts) and the stint bookkeeping
rows = []
for side in ["home", "away"]:
    t = m[["gid", "season", "week", "game_type", "gameday", "late", f"{side}_team", f"{side}_qb", f"{side}_down", f"{side}_stint3", f"{side}_538_qb_adj", "mkt_minus_noqb", "resid_noqb", "resid_base", "qb_adj_pts"]].copy()
    t.columns = ["gid", "season", "week", "game_type", "gameday", "late", "team", "qb", "down", "stint3", "adj", "mkt_gap", "resid_noqb", "resid_base", "qb_adj_pts"]
    t["side"] = side; s = 1 if side == "home" else -1
    t["market_pen"] = t.mkt_gap * s; t["realized_pen"] = -t.resid_noqb * s; t["after538_pen"] = -t.resid_base * s; t["net538_pen"] = -t.qb_adj_pts * s
    rows.append(t)
tg = pd.concat(rows); tg["team"] = tg.team.map(norm); tg["gdate"] = pd.to_datetime(tg.gameday)
tg = tg.sort_values(["team", "gdate"]).reset_index(drop=True); tg["adj_pts"] = tg.adj / 25.0
tg["prev_adj"] = tg.groupby("team").adj_pts.shift(1); tg["prev_season"] = tg.groupby("team").season.shift(1)
# last game with the displaced starter = the game just before the stint began: walk back stint3-1 games
tg["idx"] = tg.groupby("team").cumcount()
base_adj = []
for i, r in tg.iterrows():
    if r.down == 1:
        j = i - int(r.stint3)   # row of the last game before this stint (same team, since sorted by team)
        base_adj.append(tg.adj_pts.iloc[j] if j >= 0 and tg.team.iloc[j] == r.team and tg.season.iloc[j] == r.season else np.nan)
    else: base_adj.append(np.nan)
tg["starter_adj"] = base_adj
tg["A_proxy"] = tg.starter_adj - tg.adj_pts          # + = nfelo charges the team this many points vs the starter game
ev = tg[(tg.down == 1) & (tg.late == 0) & tg.A_proxy.notna()].copy()
ev["stint"] = np.where(ev.stint3 == 1, "1st", np.where(ev.stint3 <= 3, "2nd-3rd", "4th+"))
print("A_nfelo proxy = (team's 538 adj in last starter game - adj now), pts; in-season downgrade team-games with nfelo")
for st in ["1st", "2nd-3rd", "4th+"]:
    e = ev[ev.stint == st]
    print(f"  {st:<8s} n={len(e):3d}  A_proxy mean={e.A_proxy.mean():+.2f} median={e.A_proxy.median():+.2f} sd={e.A_proxy.std():.2f}  p10/p90={e.A_proxy.quantile(.1):+.2f}/{e.A_proxy.quantile(.9):+.2f}  | net538 (own-opp) pen mean={e.net538_pen.mean():+.2f}  market pen mean={e.market_pen.mean():+.2f}")
e1 = ev[ev.stint == "1st"]
r = sm.OLS(e1.market_pen, sm.add_constant(e1[["A_proxy"]])).fit(cov_type="HC1")
print(f"\n  stint-1: market penalty ~ A_proxy: intercept={r.params.const:+.2f} slope={r.params.A_proxy:+.2f}±{r.bse.A_proxy:.2f} R2={r.rsquared:.3f}  (flat-2.3 model R2=0 by construction)")
r2 = sm.OLS(e1.realized_pen, sm.add_constant(e1[["A_proxy"]])).fit(cov_type="HC1")
print(f"  stint-1: realized penalty ~ A_proxy: intercept={r2.params.const:+.2f} slope={r2.params.A_proxy:+.2f}±{r2.bse.A_proxy:.2f} (p={r2.pvalues.A_proxy:.3f})")
print(f"  share of stint-1 events where A_proxy < 1.0 pt (nfelo barely charged the backup): {(e1.A_proxy < 1).mean():.2f};  where A_proxy > 4: {(e1.A_proxy > 4).mean():.2f}")
print("  -> gap-to-target for the expert's rule at stint 1, P=2.75: mean max(0, P - A_proxy) =", round(np.maximum(0, 2.75 - e1.A_proxy).mean(), 2), "; clip(P-A,-1,P) mean =", round(np.clip(2.75 - e1.A_proxy, -1, 2.75).mean(), 2))
# does a per-game gap rule beat flat on the market target OOS? (fit nothing: rule is P - A_proxy)
for lab, e in [("<=2021", e1[e1.season <= 2021]), ("2022-25", e1[e1.season >= 2022])]:
    err_flat = e.market_pen - 2.75; err_gap = e.market_pen - e.A_proxy
    print(f"  {lab}: MAE of market penalty vs flat 2.75 = {err_flat.abs().mean():.2f}; vs A_proxy alone = {err_gap.abs().mean():.2f}; vs 2.16+0.41*A (critic_01 G) = {(e.market_pen - (2.16 + 0.41*e.A_proxy)).abs().mean():.2f}  (n={len(e)})")
