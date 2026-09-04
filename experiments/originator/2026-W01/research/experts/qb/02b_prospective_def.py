"""02b: prospective (no look-ahead) backup definition D3 + ATS rates.
D3: a stint (consecutive starts by the same QB for a team) beginning at week>=2 is a 'downgrade stint' if the new starter
had fewer career starts (1999+) than the displaced starter at the moment of the change. All games of that stint are
flagged until the QB changes again. Also 'upgrade stint' (returning/more experienced starter) as a symmetry check.
Adds net_D3, net_up to qb_games_defs.csv."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm

g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows=[]
for side in ["home","away"]:
    t=g_all[["gid","season","week","game_type","gdate",f"{side}_team",f"{side}_qb_name"]].copy()
    t.columns=["gid","season","week","game_type","gdate","team","qb"]; t["side"]=side; rows.append(t)
tg=pd.concat(rows).sort_values(["gdate","gid"]).reset_index(drop=True)
tg["qb"]=tg.qb.fillna("UNKNOWN"); tg["team"]=tg.team.map(norm)
tg["career"]=tg.groupby("qb").cumcount()          # prior career starts
tg=tg.sort_values(["team","gdate"]).reset_index(drop=True)
tg["prev_qb"]=tg.groupby("team").qb.shift(1)
tg["prev_career"]=tg.groupby("team").career.shift(1)   # displaced QB's prior starts at the time of HIS last start
flag_down=[]; flag_up=[]; stint_i=[]
for team, grp in tg.groupby("team", sort=False):
    cur=0; up=0; last=None; c=0
    for _, r in grp.iterrows():
        if r.week==1 and r.game_type=="REG":
            # season boundary: offseason changes are not 'backup' events (handled in 04_week1); reset flags + stint
            cur=0; up=0; c=1
        elif r.qb != last:
            c=1
            if last is None:
                cur=0; up=0
            else:
                cur = int(r.career < r.prev_career)      # downgrade: less experienced replaces more experienced
                up  = int(r.career > r.prev_career)
        else:
            c+=1
        flag_down.append(cur); flag_up.append(up); stint_i.append(c); last=r.qb
tg["down"]=flag_down; tg["up"]=flag_up; tg["stint3"]=stint_i
h=tg[tg.side=="home"].set_index("gid"); a=tg[tg.side=="away"].set_index("gid")
gl=pd.concat([h[["down","up","stint3"]].add_prefix("home_"), a[["down","up","stint3"]].add_prefix("away_")],axis=1).reset_index()
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False).drop(columns=[c for c in ["home_down","away_down","home_up","away_up","home_stint3","away_stint3","net_D3","net_up"] if c in pd.read_csv(HERE/"qb_games_defs.csv", nrows=1).columns])
m=m.merge(gl,on="gid",how="left")
m["net_D3"]=m.home_down-m.away_down; m["net_up"]=m.home_up-m.away_up
m.to_csv(HERE/"qb_games_defs.csv", index=False)

def ols(y,x,df,label=""):
    d=df.dropna(subset=[y,x]); r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1")
    b,se,p=r.params[x],r.bse[x],r.pvalues[x]; nb=int((d[x]!=0).sum())
    print(f"  {label:<40s} n={len(d):4d} (event games={nb:4d}) coef={b:+.3f} se={se:.3f} 95%CI=[{b-1.96*se:+.2f},{b+1.96*se:+.2f}] p={p:.4f}")
    return b,se,nb
def ats_rate(df, x, label):
    """ATS of FADING the team with the backup (bet opponent) vs closing line."""
    d=df[(df[x]!=0)&df.resid_mkt.notna()]
    fade_wins = ((d[x]==1)&(d.resid_mkt<0)) | ((d[x]==-1)&(d.resid_mkt>0))
    push = d.resid_mkt==0
    w=int(fade_wins.sum()); l=int((~fade_wins & ~push).sum()); n=w+l
    if n==0: return
    p=w/n; ci=stats.binomtest(w,n).proportion_ci(0.95); pv=stats.binomtest(w,n,0.5).pvalue
    print(f"  {label:<40s} fade-backup ATS {w}-{l}-{int(push.sum())} = {p:.3f}  95%CI=[{ci.low:.3f},{ci.high:.3f}]  p(vs .5)={pv:.4f}")

print("D3 event games (team-games, 2009+ with nfelo):", int(m.home_down.sum()+m.away_down.sum()), "| upgrade events:", int(m.home_up.sum()+m.away_up.sum()))
print("net_D3 distribution:", m.net_D3.value_counts().to_dict())
print("overlap: share of D3 backup team-games also D1:", round(((m.home_down==1)&(m.home_is_backup==1)).sum()/max(1,m.home_down.sum()),3))
for era,d in [("ALL 2009-2025",m),("2009-2015",m[m.era=="2009-2015"]),("2016-2025",m[m.era=="2016-2025"])]:
    print(f"===== D3 prospective downgrade stint — {era}")
    ols("mkt_minus_noqb","net_D3",d,"(a) market move vs no-QB line")
    ols("resid_noqb","net_D3",d,"(b) result resid vs no-QB line")
    ols("resid_base","net_D3",d,"(c) resid vs line WITH 538 adj")
    ols("resid_mkt","net_D3",d,"(d) resid vs MARKET close")
    ols("qb_adj_pts","net_D3",d,"(e) 538 adj pts assigned")
    ats_rate(d,"net_D3","(f) fade-backup ATS vs close")
print("===== D3 by stint (ALL)")
for b,(lo,hi) in [("1",(1,1)),("2-3",(2,3)),("4+",(4,99))]:
    d=m[(m.net_D3==0) | ((m.home_down==1)&m.home_stint3.between(lo,hi)) | ((m.away_down==1)&m.away_stint3.between(lo,hi))]
    d=d[~((m.home_down==1)&(m.away_down==1))]
    print(f"  stint {b}:")
    ols("mkt_minus_noqb","net_D3",d,"   (a) market move vs no-QB line")
    ols("resid_noqb","net_D3",d,"   (b) result resid vs no-QB line")
    ols("resid_base","net_D3",d,"   (c) resid vs line WITH 538 adj")
    ols("resid_mkt","net_D3",d,"   (d) resid vs market")
    ols("qb_adj_pts","net_D3",d,"   (e) 538 adj pts assigned")
    ats_rate(d,"net_D3","   (f) fade-backup ATS")
print("===== symmetry: UPGRADE stints (more-experienced QB takes over / returns), ALL")
ols("mkt_minus_noqb","net_up",m,"(a) market move vs no-QB line")
ols("resid_noqb","net_up",m,"(b) result resid vs no-QB line")
ols("resid_base","net_up",m,"(c) resid vs line WITH 538 adj")
ols("resid_mkt","net_up",m,"(d) resid vs market")
print("===== ATS for the look-ahead definitions (for comparison)")
ats_rate(m,"net_D1","D1 (season-primary) ALL"); ats_rate(m,"net_D2","D2 ALL")
ats_rate(m[m.stint_D1=="1"],"net_D1","D1 stint 1 only")
print("===== D1 look-ahead bias check: D1 backup games split by whether displaced primary later RETURNED that season is not observable prospectively;")
print("      instead compare D1-but-not-D3 games (benched/early-hurt starters, week-1 changes) vs D3 games")
d=m[(m.net_D1!=0)&(m.net_D3==0)]; print("  D1-only games:", len(d), " mean resid_mkt*sign:", round((d.resid_mkt*d.net_D1).mean(),3), " mean resid_noqb*sign:", round((d.resid_noqb*d.net_D1).mean(),3))
d=m[(m.net_D3!=0)]; print("  D3 games:", len(d), " mean resid_mkt*sign:", round((d.resid_mkt*d.net_D3).mean(),3), " mean resid_noqb*sign:", round((d.resid_noqb*d.net_D3).mean(),3))
