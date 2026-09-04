"""CRITIC 06: loose ends. (a) wk17+ backup starts on ELIMINATED teams (from critic_03) by fit/test window; (b) stint-1 in-season
events split by a benching proxy (displaced starter's own 538 adj <= -1, i.e. he was rated well below the team baseline) vs
likely-injury (adj > -1): does the market over-penalise benchings?  (c) 4th+ stint realized penalty (is 0.5 too low?), fit/test.
Re-runnable."""
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
m["s4"] = ((m.home_down == 1) & (m.home_stint3 >= 4)).astype(int) - ((m.away_down == 1) & (m.away_stint3 >= 4)).astype(int)
def coef(y, x, d):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())
# team-game with previous game's own adj
rows = []
for side in ["home", "away"]:
    t = m[["gid", "season", "gameday", f"{side}_team", f"{side}_538_qb_adj", f"{side}_down", f"{side}_stint3"]].copy()
    t.columns = ["gid", "season", "gameday", "team", "adj", "down", "stint3"]; t["side"] = side; rows.append(t)
tg = pd.concat(rows); tg["team"] = tg.team.map(norm); tg["gdate"] = pd.to_datetime(tg.gameday)
tg = tg.sort_values(["team", "gdate"]).reset_index(drop=True); tg["prev_adj"] = tg.groupby("team").adj.shift(1) / 25.0
pa = {s: tg[tg.side == s].set_index("gid").prev_adj for s in ["home", "away"]}
ev = m[~both & (m.late == 0) & (m.s1 != 0)].copy()
ev["prev_adj"] = np.where(ev.s1 == 1, ev.gid.map(pa["home"]), ev.gid.map(pa["away"]))
ev["market"] = ev.mkt_minus_noqb * ev.s1; ev["realized"] = -ev.resid_noqb * ev.s1; ev["vs_mkt"] = -ev.resid_mkt * ev.s1; ev["after538"] = -ev.resid_base * ev.s1
print("(b) stint-1 in-season events split by benching proxy (displaced starter's prior-game 538 adj):")
for lab, mask in [("benching proxy: adj <= -1", ev.prev_adj <= -1), ("likely injury: adj > -1", ev.prev_adj > -1)]:
    for era, e in [("ALL", ev[mask]), ("<=2021", ev[mask & (ev.season <= 2021)]), ("2022-25", ev[mask & (ev.season >= 2022)])]:
        f = lambda c: f"{e[c].mean():+.2f}±{e[c].std(ddof=1)/np.sqrt(len(e)):.2f}"
        print(f"  {lab:<28s} {era:<8s} n={len(e):3d} market={f('market')} realized={f('realized')} after538={f('after538')} vs_mkt={f('vs_mkt')} median vs_mkt={e.vs_mkt.median():+.1f}")
print("\n(c) 4th+ stint realized penalty vs no-QB line, in-season, opponent not a backup (expert table says 0.5):")
for era, d in [("ALL", m), ("<=2021", m[m.season <= 2021]), ("2022-25", m[m.season >= 2022])]:
    dd = d[~both & (d.late == 0) & ((d.s4 != 0) | (d.net_D3 == 0))]
    out = []
    for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("qb_adj_pts", "538adj", -1), ("resid_base", "after538", -1), ("resid_mkt", "vs_mkt", -1)]:
        b, se, p, n = coef(y, "s4", dd); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}")
    print(f"  {era:<8s} n_event={n:3d}  " + "  ".join(out))
print("\n(a) wk17/18 stint-1 backup on a team that did NOT make the playoffs (critic_03 subgroup), by window:")
g_all = load_games(1999)
po = g_all[g_all.game_type != "REG"]
made = set(zip(po.season, po.home_team.map(norm))) | set(zip(po.season, po.away_team.map(norm)))
late = m[~both & (m.late == 1) & (m.game_type == "REG") & (m.s1 != 0)].copy()
late["team"] = np.where(late.s1 == 1, late.home_team.map(norm), late.away_team.map(norm))
late["made_po"] = [(s, t) in made for s, t in zip(late.season, late.team)]
late["market"] = late.mkt_minus_noqb * late.s1; late["realized"] = -late.resid_noqb * late.s1; late["vs_mkt"] = -late.resid_mkt * late.s1
for lab, e in [("non-playoff team, ALL", late[~late.made_po]), ("non-playoff team, <=2021", late[~late.made_po & (late.season <= 2021)]), ("non-playoff team, 2022-25", late[~late.made_po & (late.season >= 2022)])]:
    f = lambda c: f"{e[c].mean():+.2f}±{e[c].std(ddof=1)/np.sqrt(len(e)):.2f}"
    print(f"  {lab:<28s} n={len(e):2d} market={f('market')} realized={f('realized')} vs_mkt={f('vs_mkt')} median vs_mkt={e.vs_mkt.median():+.1f}")
