"""CRITIC 03 (theory qb-2b): the 59 'rest' games (wk17+/playoffs, stint-1 downgrade) are NOT all rest games.
Decompose them prospectively-unobservable-but-diagnostic: playoff games (injury by definition), REG wk17/18 where the team
then played a playoff game AND the displaced starter started it (true rest), REG wk17/18 where the team made the playoffs
but the starter did NOT return for the playoff game (injury), and REG wk17/18 for non-playoff teams (eliminated: injury /
benching / evaluation).  Report market-implied, realized, 538, after-538, vs-market for each subgroup.  Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna()].copy()
m["late"] = ((m.game_type != "REG") | (m.week >= 17)).astype(int)
both = (m.home_down == 1) & (m.away_down == 1)
m["s1"] = ((m.home_down == 1) & (m.home_stint3 == 1)).astype(int) - ((m.away_down == 1) & (m.away_stint3 == 1)).astype(int)
ev = m[~both & (m.late == 1) & (m.s1 != 0)].copy()
print("late-season stint-1 downgrade events (clean):", len(ev), " by game_type/week:", ev.groupby(["game_type", "week"]).size().to_dict())
# team-game history (all games incl. playoffs) to find the team's next game and its starter
g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows = []
for side in ["home", "away"]:
    t = g_all[["gid", "season", "week", "game_type", "gdate", f"{side}_team", f"{side}_qb_name", "result"]].copy()
    t.columns = ["gid", "season", "week", "game_type", "gdate", "team", "qb", "result"]; t["side"] = side; rows.append(t)
tg = pd.concat(rows).sort_values(["team", "gdate"]).reset_index(drop=True); tg["team"] = tg.team.map(norm)
tg["prev_qb"] = tg.groupby("team").qb.shift(1); tg["next_qb"] = tg.groupby("team").qb.shift(-1)
tg["next_type"] = tg.groupby("team").game_type.shift(-1); tg["next_season"] = tg.groupby("team").season.shift(-1)
tg["prev_season"] = tg.groupby("team").season.shift(1)
made_po = tg[tg.game_type != "REG"].groupby(["season", "team"]).size().rename("po_games").reset_index()
tg = tg.merge(made_po, on=["season", "team"], how="left"); tg["po_games"] = tg.po_games.fillna(0)
key = tg.set_index(["gid", "side"])
rows = []
for _, r in ev.iterrows():
    side = "home" if r.s1 == 1 else "away"; t = key.loc[(r.gid, side)]
    sgn = r.s1
    rec = dict(gid=r.gid, season=r.season, week=r.week, gtype=r.game_type, team=t.team, backup=t.qb, displaced=t.prev_qb,
               made_playoffs=int(t.po_games > 0), next_is_playoff=int((t.next_type != "REG") & (t.next_season == r.season)) if isinstance(t.next_type, str) else 0,
               starter_returned_next=int(t.next_qb == t.prev_qb) if isinstance(t.next_qb, str) else 0,
               mkt_line_for_team=-(r.mkt_spread * sgn),  # + = team favored by that many
               market=r.mkt_minus_noqb * sgn, realized=-r.resid_noqb * sgn, adj538=-r.qb_adj_pts * sgn, after538=-r.resid_base * sgn, vs_mkt=-r.resid_mkt * sgn)
    rows.append(rec)
E = pd.DataFrame(rows)
def cls(r):
    if r.gtype != "REG": return "playoff game (injury by definition)"
    if r.made_playoffs and r.next_is_playoff and r.starter_returned_next: return "REG wk17+, TRUE REST (starter started next playoff game)"
    if r.made_playoffs: return "REG wk17+, playoff team but starter did NOT return next game"
    return "REG wk17+, non-playoff team (eliminated / injury / eval)"
E["class"] = E.apply(cls, axis=1)
def block(lab, e):
    if len(e) == 0: print(f"  {lab}: n=0"); return
    f = lambda c: f"{e[c].mean():+.2f}±{e[c].std(ddof=1)/np.sqrt(len(e)) if len(e)>1 else float('nan'):.2f}"
    print(f"  {lab:<62s} n={len(e):2d}  market={f('market')}  realized={f('realized')}  538={f('adj538')}  after538={f('after538')}  vs_mkt={f('vs_mkt')}  median vs_mkt={e.vs_mkt.median():+.1f}  team favored by (mkt) {e.mkt_line_for_team.mean():+.1f}")
print("\nSubgroups (penalty-signed, + = backup team penalised / underperformed):")
block("ALL late-season stint-1 (expert's n=59)", E)
for c in ["REG wk17+, TRUE REST (starter started next playoff game)", "REG wk17+, playoff team but starter did NOT return next game", "REG wk17+, non-playoff team (eliminated / injury / eval)", "playoff game (injury by definition)"]:
    block(c, E[E["class"] == c])
print("\n  fit(<=2021)/test(2022-25) split of the TRUE-REST subgroup:")
tr = E[(E["class"].str.contains("TRUE REST")) & (E.season <= 2021)]; te = E[(E["class"].str.contains("TRUE REST")) & (E.season >= 2022)]
block("TRUE REST, 2009-2021", tr); block("TRUE REST, 2022-2025", te)
print("\n  TRUE-REST game list:")
print(E[E["class"].str.contains("TRUE REST")][["season", "week", "team", "displaced", "backup", "mkt_line_for_team", "market", "realized", "vs_mkt"]].round(1).to_string(index=False))
print("\n  non-playoff-team wk17+ list (first 15):")
print(E[E["class"].str.contains("non-playoff")][["season", "week", "team", "displaced", "backup", "mkt_line_for_team", "market", "realized", "vs_mkt"]].round(1).head(15).to_string(index=False))
# Also: ALL wk17+ games where a team's week-17/18 starter differs from its previous-game starter (any experience direction), rest proxy
print("\nBroader rest proxy (not restricted to D3 downgrade): REG wk17+ team-games where the starter differs from the previous game's starter,")
print("team made playoffs, and the previous starter started the next playoff game:")
tg2 = tg[(tg.season >= 2009) & (tg.game_type == "REG") & (tg.week >= 17) & (tg.qb != tg.prev_qb) & (tg.po_games > 0)]
tg2 = tg2[(tg2.next_type != "REG") & (tg2.next_qb == tg2.prev_qb)]
hm = tg2[tg2.side == "home"].set_index("gid").index; am = tg2[tg2.side == "away"].set_index("gid").index
m["rest_net"] = m.gid.isin(hm).astype(int) - m.gid.isin(am).astype(int)
d = m[(m.late == 1) & (m.game_type == "REG")]
def coef(y, x, d):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())
out = []
for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("qb_adj_pts", "538adj", -1), ("resid_base", "after538", -1), ("resid_mkt", "vs_mkt", -1)]:
    b, se, p, n = coef(y, "rest_net", d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}(p={p:.2f})")
print(f"  n_event={n}  " + "  ".join(out))
for lab, dd in [("<=2021", d[d.season <= 2021]), ("2022-25", d[d.season >= 2022])]:
    out = []
    for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("resid_mkt", "vs_mkt", -1)]:
        b, se, p, n = coef(y, "rest_net", dd); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}(p={p:.2f})")
    print(f"    {lab}: n_event={n}  " + "  ".join(out))
