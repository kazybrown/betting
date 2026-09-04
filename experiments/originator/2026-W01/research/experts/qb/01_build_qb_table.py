"""01: build team-game QB starter table + game-level backup flags, merged with nfelo.
Writes qb_games.csv next to this script (re-runnable). Prints coverage stats."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, load_nfelo
OUT = Path(__file__).resolve().parent

# full history from 1999 for career-start counts; analysis window 2009+
g_all = load_games(1999)
g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows = []
for side, opp in [("home", "away"), ("away", "home")]:
    t = g_all[["game_id","gid","season","game_type","week","gdate",f"{side}_team",f"{side}_qb_name",f"{side}_qb_id"]].copy()
    t.columns = ["game_id","gid","season","game_type","week","gdate","team","qb","qb_id"]
    t["side"] = side
    rows.append(t)
tg = pd.concat(rows).sort_values(["gdate","game_id"]).reset_index(drop=True)
from kit import norm
tg["team"] = tg.team.map(norm)
tg["qb"] = tg.qb.fillna("UNKNOWN")

# career prior starts (any team, 1999+), by qb name
tg["career_prior_starts"] = tg.groupby("qb").cumcount()
# team-season primary starter = most REG-season starts (tie -> earliest first start)
reg = tg[tg.game_type=="REG"]
cnt = reg.groupby(["season","team","qb"]).agg(n=("qb","size"), first=("gdate","min")).reset_index()
cnt = cnt.sort_values(["season","team","n","first"], ascending=[True,True,False,True])
primary = cnt.drop_duplicates(["season","team"])[["season","team","qb","n"]].rename(columns={"qb":"primary_qb","n":"primary_starts"})
tg = tg.merge(primary, on=["season","team"], how="left")
tg["is_backup"] = (tg.qb != tg.primary_qb).astype(int)
# previous game's starter for the team (chronological, across seasons)
tg = tg.sort_values(["team","gdate"]).reset_index(drop=True)
tg["prev_qb"] = tg.groupby("team").qb.shift(1)
tg["prev_season"] = tg.groupby("team").season.shift(1)
tg["qb_changed"] = (tg.qb != tg.prev_qb).astype(int)
# stint index: consecutive starts by this QB for this team (resets on change, not on season boundary)
stint = []
for team, grp in tg.groupby("team", sort=False):
    c = 0; last = None
    for q in grp.qb:
        c = c+1 if q == last else 1
        stint.append(c); last = q
    # note: grp preserved order because tg sorted by team,gdate
tg["stint_idx"] = stint
# starts this season for this QB with this team, before this game
tg["season_prior_starts_team"] = tg.groupby(["season","team","qb"]).cumcount()
# previous season primary
prev_primary = primary.rename(columns={"primary_qb":"prev_season_primary"}).assign(season=lambda d: d.season+1)[["season","team","prev_season_primary"]]
tg = tg.merge(prev_primary, on=["season","team"], how="left")

# game level
h = tg[tg.side=="home"].set_index("gid"); a = tg[tg.side=="away"].set_index("gid")
cols = ["qb","primary_qb","is_backup","prev_qb","qb_changed","stint_idx","career_prior_starts","season_prior_starts_team","prev_season_primary","primary_starts"]
gl = pd.concat([h[cols].add_prefix("home_"), a[cols].add_prefix("away_")], axis=1).reset_index()
g = load_games(2009).merge(gl, on="gid", how="left")
n = load_nfelo()
n["nfelo_dif_noqb"] = n.nfelo_dif_base - n.home_net_qb_mod
n["line_nfelo_base"] = -n.nfelo_dif_base/25.0            # ORIGINATOR convention, unregressed, WITH 538 QB adj
n["line_nfelo_noqb"] = -n.nfelo_dif_noqb/25.0            # same WITHOUT QB adj
n["qb_adj_pts"] = n.home_net_qb_mod/25.0                  # + = favors home
keep = ["gid","starting_nfelo_home","starting_nfelo_away","hfa_base_mod","home_538_qb_adj","away_538_qb_adj","home_net_qb_mod",
        "nfelo_dif_base","nfelo_dif_noqb","nfelo_dif_close","line_nfelo_base","line_nfelo_noqb","qb_adj_pts",
        "nfelo_home_line_open","nfelo_home_line_close","home_line_open","home_line_close","total_line_close"]
m = g.merge(n[keep], on="gid", how="left")
m["backup_net"] = m.home_is_backup - m.away_is_backup          # +1 home backup, -1 away backup
m["changed_net"] = m.home_qb_changed - m.away_qb_changed
m["resid_mkt"]  = m.margin + m.mkt_spread                      # margin residual vs market close (>0 home beat number)
m["resid_noqb"] = m.margin + m.line_nfelo_noqb                 # residual vs nfelo unregressed line w/o QB adj
m["resid_base"] = m.margin + m.line_nfelo_base                 # residual vs nfelo unregressed line WITH QB adj
m["mkt_minus_noqb"] = m.mkt_spread - m.line_nfelo_noqb         # market movement relative to no-QB rating line (+ = market likes home LESS)
m["era"] = np.where(m.season<=2015, "2009-2015", "2016-2025")
m.to_csv(OUT/"qb_games.csv", index=False)

print("games 2009+:", len(m), " with nfelo:", m.line_nfelo_noqb.notna().sum(), " with market:", m.mkt_spread.notna().sum())
print("backup starts (team-games, REG 2009+):", int(m[m.game_type=='REG'][['home_is_backup','away_is_backup']].sum().sum()),
      "of", 2*len(m[m.game_type=='REG']))
print("share of team-games started by backup by season:")
print(m[m.game_type=='REG'].groupby('season').apply(lambda d: (d.home_is_backup.sum()+d.away_is_backup.sum())/(2*len(d))).round(3).to_dict())
print("backup_net distribution:", m.backup_net.value_counts().to_dict())
print("home stint_idx describe for backups:"); print(m.loc[m.home_is_backup==1,"home_stint_idx"].describe().round(2))
# nfelo line-vs-Elo linearity
sub = m.dropna(subset=["nfelo_home_line_close","nfelo_dif_close"])
b = np.polyfit(-sub.nfelo_dif_close/25, sub.nfelo_home_line_close, 1)
print("nfelo_home_line_close ~ (-nfelo_dif_close/25): slope=%.3f intercept=%.3f" % (b[0], b[1]))
# 538 QB adj on backup games
print("\nmean qb_adj_pts (home-favoring points) by backup_net:")
print(m.groupby("backup_net").qb_adj_pts.agg(["mean","median","count"]).round(3))
print("wrote", OUT/"qb_games.csv")
