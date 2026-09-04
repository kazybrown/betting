"""CRITIC 01b (theory qb-1): is the market-implied backup penalty (market close vs no-QB nfelo line) contaminated by a
team-level 'market dislikes this team more than the Elo does' component?  Look at the SAME teams 1, 2, 3 games BEFORE the
stint-1 downgrade (starter still playing) and at the game the displaced starter RETURNS (first upgrade start after a
downgrade stint).  Also decompose the week-before game by whether the change was a likely injury (team lost by a lot with
the starter) vs a benching.  Uses the expert's name-keyed team-game logic.  Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna()].copy()
m["late"] = ((m.game_type != "REG") | (m.week >= 17)).astype(int)

g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows = []
for side in ["home", "away"]:
    t = g_all[["gid", "season", "week", "game_type", "gdate", f"{side}_team", f"{side}_qb_name"]].copy()
    t.columns = ["gid", "season", "week", "game_type", "gdate", "team", "qb"]; t["side"] = side; rows.append(t)
tg = pd.concat(rows).sort_values(["gdate", "gid"]).reset_index(drop=True)
tg["team"] = tg.team.map(norm); tg["career"] = tg.groupby("qb").cumcount()
tg = tg.sort_values(["team", "gdate"]).reset_index(drop=True)
tg["prev_career"] = tg.groupby("team").career.shift(1)
down = []; up = []; st = []
for team, grp in tg.groupby("team", sort=False):
    cur = 0; u = 0; last = None; c = 0
    for _, r in grp.iterrows():
        if r.week == 1 and r.game_type == "REG": cur = 0; u = 0; c = 1
        elif r.qb != last:
            c = 1
            if last is None: cur = 0; u = 0
            else: cur = int(r.career < r.prev_career); u = int(r.career > r.prev_career)
        else: c += 1
        down.append(cur); up.append(u); st.append(c); last = r.qb
tg["down"] = down; tg["up"] = up; tg["stint3"] = st
tg = tg[tg.season >= 2009].sort_values(["team", "gdate"]).reset_index(drop=True)
tg["ev1"] = ((tg.down == 1) & (tg.stint3 == 1)).astype(int)
# lags: game k BEFORE a stint-1 downgrade, same season, starter (non-downgrade) still playing
for k in [1, 2, 3]:
    nxt = tg.groupby("team").ev1.shift(-k); nseason = tg.groupby("team").season.shift(-k)
    tg[f"pre{k}"] = ((nxt == 1) & (nseason == tg.season) & (tg.down == 0)).astype(int)
# starter return: first upgrade start immediately following a downgrade stint (previous game was a downgrade game)
prev_down = tg.groupby("team").down.shift(1); prev_season = tg.groupby("team").season.shift(1)
tg["ret"] = ((tg.up == 1) & (tg.stint3 == 1) & (prev_down == 1) & (prev_season == tg.season)).astype(int)
# margin in the week-before game for the team (sign by side)
def side_map(col):
    h = tg[tg.side == "home"].set_index("gid")[col]; a = tg[tg.side == "away"].set_index("gid")[col]
    return m.gid.map(h).fillna(0).astype(int) - m.gid.map(a).fillna(0).astype(int)
for c in ["pre1", "pre2", "pre3", "ret", "ev1"]:
    m[f"net_{c}"] = side_map(c)
def coef(y, x, d):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1")
    return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())
print("sign check corr(mkt_spread, margin) =", round(np.corrcoef(m.mkt_spread, m.margin)[0, 1], 3))
print("Penalty-signed coefficients (+ = market/result against the flagged team), games in-season where neither side is in a downgrade stint")
base = m[(m.late == 0) & (m.net_D3 == 0)]
for c, lab in [("pre3", "3 games before the change"), ("pre2", "2 games before the change"), ("pre1", "1 game before (starter's last game)")]:
    out = []
    for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("resid_mkt", "vs_mkt", -1), ("qb_adj_pts", "538adj", -1)]:
        b, se, p, n = coef(y, f"net_{c}", base); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}")
    print(f"  {lab:<38s} n_event={n:3d}  " + "  ".join(out))
print("  -> stint-1 event itself (for comparison, in-season, opponent not a backup):")
both = (m.home_down == 1) & (m.away_down == 1)
d = m[~both & (m.late == 0) & ((m.net_ev1 != 0) | (m.net_D3 == 0))]
out = []
for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("resid_mkt", "vs_mkt", -1), ("qb_adj_pts", "538adj", -1)]:
    b, se, p, n = coef(y, "net_ev1", d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}")
print(f"  {'stint-1 downgrade game':<38s} n_event={n:3d}  " + "  ".join(out))
print("\nStarter RETURN game (first upgrade start right after a downgrade stint), in-season; if the Elo has absorbed the backup, the market should")
print("like the returning team MORE than the no-QB line (negative 'penalty'):")
d = m[(m.late == 0) & (m.net_D3 == 0)]
out = []
for y, nm, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("resid_mkt", "vs_mkt", -1), ("qb_adj_pts", "538adj", -1)]:
    b, se, p, n = coef(y, "net_ret", d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}(p={p:.2f})")
print(f"  {'starter-return game':<38s} n_event={n:3d}  " + "  ".join(out))
# week-before decomposition: did the team lose badly in the starter's last game (injury-in-game proxy) or not (benching proxy)?
print("\nWeek-before game split by the flagged team's result in that game (proxy: in-game injury/blowout vs benching after a normal result):")
sgn = m.net_pre1
pre = m[(m.late == 0) & (m.net_D3 == 0) & (sgn != 0)].copy()
pre["team_margin"] = pre.margin * pre.net_pre1; pre["mkt_pen"] = pre.mkt_minus_noqb * pre.net_pre1; pre["real_pen"] = -pre.resid_noqb * pre.net_pre1
for lab, mask in [("team lost by 14+", pre.team_margin <= -14), ("lost by <14", (pre.team_margin < 0) & (pre.team_margin > -14)), ("won", pre.team_margin > 0)]:
    e = pre[mask]; print(f"  {lab:<18s} n={len(e):3d}  market-vs-noQB in that game={e.mkt_pen.mean():+.2f}±{e.mkt_pen.std()/np.sqrt(len(e)):.2f}   realized={e.real_pen.mean():+.2f}")
# then: the stint-1 market penalty conditional on the week-before market gap (does the +0.7 persist into the event week?)
tg["pre1_gid"] = tg.groupby("team").gid.shift(1)
ev = tg[(tg.ev1 == 1)].copy()
gap = m.set_index("gid")
def side_gap(gid, side):
    if gid not in gap.index: return np.nan
    r = gap.loc[gid]; s = 1 if side == "home" else -1; return r.mkt_minus_noqb * s
prev_side = tg.set_index(["gid", "team"]).side
ev["pre_gap"] = [side_gap(pg, prev_side.get((pg, t), None)) if isinstance(pg, str) else np.nan for pg, t in zip(ev.pre1_gid, ev.team)]
ev["ev_gap"] = [side_gap(g_, s) for g_, s in zip(ev.gid, ev.side)]
e = ev.dropna(subset=["pre_gap", "ev_gap"])
e = e[e.gid.isin(m[(m.late == 0)].gid)]
r = sm.OLS(e.ev_gap, sm.add_constant(e[["pre_gap"]])).fit(cov_type="HC1")
print(f"\nStint-1 market gap ~ week-before market gap (same team): intercept={r.params.const:+.2f}±{r.bse.const:.2f} slope={r.params.pre_gap:+.2f}±{r.bse.pre_gap:.2f} n={len(e)}")
print(f"  mean week-before gap among stint-1 events = {e.pre_gap.mean():+.2f}; mean stint-1 gap = {e.ev_gap.mean():+.2f}; stint-1 gap net of week-before gap = {(e.ev_gap - e.pre_gap).mean():+.2f}±{(e.ev_gap - e.pre_gap).std()/np.sqrt(len(e)):.2f}")
