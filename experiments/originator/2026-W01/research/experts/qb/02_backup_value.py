"""02: Theory 1+2 — points value of a backup QB start.
(a) market close movement vs nfelo no-QB rating line; (b) realized residual vs no-QB line;
(c) residual vs nfelo line WITH 538 QB adj (theory 2); (d) residual vs market close.
All full-sample descriptive (2009-2025, labeled) with era + stint splits; OOS tests are in 03."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE/"qb_games.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna()].copy()

# --- definitions ---------------------------------------------------------
# D1 (task): starter != team-season primary (most REG starts)
# D2: D1 AND starter has fewer career starts than the primary had at the time (experience-filtered)
for s in ["home","away"]:
    prim_career = {}
    # career starts of the primary QB at the time of each game: approximate with primary's career starts in that game's season start
    m[f"{s}_backup_D1"] = m[f"{s}_is_backup"]
# to get primary's career starts, rebuild from the team-game table quickly
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm
g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows=[]
for side in ["home","away"]:
    t=g_all[["gid","season","gdate",f"{side}_team",f"{side}_qb_name"]].copy(); t.columns=["gid","season","gdate","team","qb"]; rows.append(t)
tg=pd.concat(rows).sort_values(["gdate","gid"]); tg["qb"]=tg.qb.fillna("UNKNOWN"); tg["team"]=tg.team.map(norm)
tg["career"]=tg.groupby("qb").cumcount()
# career starts of each QB as of first game of each season
first_of_season = tg.sort_values("gdate").groupby(["season","qb"]).career.min().rename("career_at_season_start").reset_index()
for s in ["home","away"]:
    m = m.merge(first_of_season.rename(columns={"qb":f"{s}_primary_qb","career_at_season_start":f"{s}_primary_career"}), on=["season",f"{s}_primary_qb"], how="left")
    m[f"{s}_backup_D2"] = ((m[f"{s}_is_backup"]==1) & (m[f"{s}_career_prior_starts"] < m[f"{s}_primary_career"].fillna(0))).astype(int)
    m[f"{s}_backup_D1"] = m[f"{s}_is_backup"]
m["net_D1"] = m.home_backup_D1 - m.away_backup_D1
m["net_D2"] = m.home_backup_D2 - m.away_backup_D2
# stint bucket of the backup (for the side that is a backup); for D-splits use min stint of backup sides
def stint_bucket(r, col):
    s = []
    if r[f"home_backup_{col}"]==1: s.append(r.home_stint_idx)
    if r[f"away_backup_{col}"]==1: s.append(r.away_stint_idx)
    if not s: return "none"
    x = min(s); return "1" if x==1 else ("2-3" if x<=3 else "4+")
for col in ["D1","D2"]:
    m[f"stint_{col}"] = m.apply(lambda r: stint_bucket(r,col), axis=1)
m["late_rest_wk"] = ((m.game_type=="REG") & (m.week>=17)).astype(int)
m.to_csv(HERE/"qb_games_defs.csv", index=False)

def ols(y, x, df, ctrl=None, label=""):
    d = df.dropna(subset=[y,x]+([ctrl] if ctrl else []))
    X = d[[x]+([ctrl] if ctrl else [])]; X = sm.add_constant(X)
    r = sm.OLS(d[y], X).fit(cov_type="HC1")
    b, se, p = r.params[x], r.bse[x], r.pvalues[x]
    nb = int((d[x]!=0).sum())
    print(f"  {label:<44s} n={len(d):4d} (backup games={nb:4d})  coef={b:+.3f}  se={se:.3f}  95%CI=[{b-1.96*se:+.2f},{b+1.96*se:+.2f}]  p={p:.4f}")
    return b, se, p, len(d), nb

print("Sign check: corr(mkt_spread, margin) =", round(np.corrcoef(m.mkt_spread, m.margin)[0,1],3))
print("Outcome variables (home perspective, points):")
print("  mkt_minus_noqb = mkt_spread - line_nfelo_noqb : + means market likes HOME LESS than no-QB rating line")
print("  resid_noqb = margin + line_nfelo_noqb : + means home beat the no-QB rating line")
print("  resid_base = margin + line_nfelo_base : residual after nfelo's 538 QB adj (theory 2)")
print("  resid_mkt  = margin + mkt_spread      : residual vs market close")
print("  x = backup_net (+1 home backup, -1 away backup): coef = points effect of HOME team having the backup\n")

results = {}
for col in ["D1","D2"]:
    x = f"net_{col}"
    print(f"===== definition {col} ({'task: starter != season primary' if col=='D1' else 'D1 AND starter has fewer career starts than primary'}) =====")
    for era, d in [("ALL 2009-2025", m), ("2009-2015", m[m.era=="2009-2015"]), ("2016-2025", m[m.era=="2016-2025"])]:
        print(f"-- {era}")
        results[(col,era,"a_mkt_move")] = ols("mkt_minus_noqb", x, d, label="(a) market move vs no-QB line")
        ols("mkt_minus_noqb", x, d, ctrl="line_nfelo_noqb", label="(a) ... controlling for line level")
        results[(col,era,"b_resid_noqb")] = ols("resid_noqb", x, d, label="(b) result resid vs no-QB line")
        results[(col,era,"c_resid_base")] = ols("resid_base", x, d, label="(c) result resid vs line WITH 538 QB adj")
        results[(col,era,"d_resid_mkt")]  = ols("resid_mkt", x, d, label="(d) result resid vs MARKET close")
        ols("qb_adj_pts", x, d, label="(e) 538 QB adj pts assigned (home-favoring)")
    print("-- by stint of the backup (ALL seasons), excluding games where both sides backup")
    for b in ["1","2-3","4+"]:
        d = m[(m[f"stint_{col}"]==b) | (m[x]==0)]
        d = d[d[x].abs()<=1]
        print(f"   stint {b}:")
        ols("mkt_minus_noqb", x, d, label="     (a) market move vs no-QB line")
        ols("resid_noqb", x, d, label="     (b) result resid vs no-QB line")
        ols("resid_base", x, d, label="     (c) resid vs line WITH 538 adj")
        ols("resid_mkt", x, d, label="     (d) resid vs market")
        ols("qb_adj_pts", x, d, label="     (e) 538 adj pts assigned")
    print("-- excluding weeks 17+ (rest games) and playoffs, ALL seasons")
    d = m[(m.game_type=="REG") & (m.week<=16)]
    ols("mkt_minus_noqb", x, d, label="   (a) market move vs no-QB line")
    ols("resid_noqb", x, d, label="   (b) result resid vs no-QB line")
    ols("resid_base", x, d, label="   (c) resid vs line WITH 538 adj")
    ols("resid_mkt", x, d, label="   (d) resid vs market")
    print()

# raw means, D1, for transparency
print("Raw mean residuals by net_D1 (ALL):")
print(m.groupby("net_D1")[["mkt_minus_noqb","resid_noqb","resid_base","resid_mkt","qb_adj_pts"]].agg(["mean","count"]).round(3))
# separate home-backup and away-backup asymmetry check
print("\nAsymmetry check (D1): home backup only vs away backup only, mean resid_mkt and mkt_minus_noqb")
print(m[(m.home_backup_D1==1)&(m.away_backup_D1==0)][["mkt_minus_noqb","resid_noqb","resid_mkt"]].mean().round(3).to_dict(), "n=", int(((m.home_backup_D1==1)&(m.away_backup_D1==0)).sum()))
print(m[(m.home_backup_D1==0)&(m.away_backup_D1==1)][["mkt_minus_noqb","resid_noqb","resid_mkt"]].mean().round(3).to_dict(), "n=", int(((m.home_backup_D1==0)&(m.away_backup_D1==1)).sum()))
