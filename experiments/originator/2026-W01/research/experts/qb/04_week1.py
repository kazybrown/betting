"""04: Theory 3 — Week 1 new-starter and first-career-start (rookie proxy) effects vs market and vs no-QB rating line.
new starter = week-1 starter != team's previous-season primary (most REG starts). first-career-start = 0 prior starts in
games.csv (1999+): a rookie or a never-started veteran (no draft data locally). Extensions: weeks 1-4 of the new
starter's tenure, and an all-weeks 'inexperienced QB (<10 career starts)' check. Small n -> mostly INCONCLUSIVE."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.mkt_spread.notna()].copy()
for s in ["home","away"]:
    m[f"{s}_new"]=(m[f"{s}_qb"]!=m[f"{s}_prev_season_primary"]).astype(int)
    m[f"{s}_first"]=(m[f"{s}_career_prior_starts"]==0).astype(int)
    m[f"{s}_inexp"]=(m[f"{s}_career_prior_starts"]<10).astype(int)
    m[f"{s}_newvet"]=((m[f"{s}_new"]==1)&(m[f"{s}_career_prior_starts"]>=16)).astype(int)   # experienced QB new to team (or returning)
    m[f"{s}_newyoung"]=((m[f"{s}_new"]==1)&(m[f"{s}_career_prior_starts"]<16)).astype(int)
for f in ["new","first","inexp","newvet","newyoung"]:
    m[f"net_{f}"]=m[f"home_{f}"]-m[f"away_{f}"]
def ols(y,x,df,label):
    d=df.dropna(subset=[y,x]); 
    if len(d)<20 or (d[x]!=0).sum()<5: print(f"  {label:<46s} n={len(d)} too few events"); return
    r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); b,se,p=r.params[x],r.bse[x],r.pvalues[x]
    print(f"  {label:<46s} n={len(d):4d} (events={int((d[x]!=0).sum()):3d}) coef={b:+.3f} se={se:.3f} 95%CI=[{b-1.96*se:+.2f},{b+1.96*se:+.2f}] p={p:.4f}")
def ats_fade(df,x,label):
    d=df[(df[x]!=0)&df.resid_mkt.notna()]; win=((d[x]==1)&(d.resid_mkt<0))|((d[x]==-1)&(d.resid_mkt>0)); push=d.resid_mkt==0
    w=int(win.sum()); l=int((~win&~push).sum()); n=w+l
    if n<5: return
    ci=stats.binomtest(w,n).proportion_ci(0.95); print(f"  {label:<46s} fade ATS {w}-{l}-{int(push.sum())} = {w/n:.3f} 95%CI=[{ci.low:.3f},{ci.high:.3f}] p={stats.binomtest(w,n,0.5).pvalue:.3f}")

w1=m[(m.game_type=="REG")&(m.week==1)]
print(f"Week 1 games 2009-2025: {len(w1)} (with nfelo: {int(w1.line_nfelo_noqb.notna().sum())}); team-games with new starter: {int(w1.home_new.sum()+w1.away_new.sum())}, first-career-start: {int(w1.home_first.sum()+w1.away_first.sum())}")
print("sign check corr(mkt_spread, margin) week1:", round(np.corrcoef(w1.mkt_spread,w1.margin)[0,1],3))
for f,lab in [("new","new starter vs prev-season primary"),("first","first career start (rookie proxy)"),("newyoung","new starter, <16 career starts"),("newvet","new starter, >=16 career starts")]:
    print(f"--- Week 1: {lab}")
    ols("resid_mkt",f"net_{f}",w1,"resid vs MARKET close")
    ols("mkt_minus_noqb",f"net_{f}",w1,"market move vs no-QB nfelo line")
    ols("resid_noqb",f"net_{f}",w1,"resid vs no-QB nfelo line")
    ols("resid_base",f"net_{f}",w1,"resid vs nfelo line WITH 538 adj")
    ols("qb_adj_pts",f"net_{f}",w1,"538 adj pts assigned")
    ats_fade(w1,f"net_{f}","fade-new-starter ATS")
    for era in ["2009-2015","2016-2025"]:
        ols("resid_mkt",f"net_{f}",w1[w1.era==era],f"  resid vs market, {era}")
# split sample by fit/test for the market residual (discipline)
print("--- Week 1 resid vs market, fit (<=2021) vs test (2022-2025)")
for f in ["new","first"]:
    ols("resid_mkt",f"net_{f}",w1[w1.season<=2021],f"{f}: <=2021"); ols("resid_mkt",f"net_{f}",w1[w1.season>=2022],f"{f}: 2022-2025")
# weeks 1-4 of a new starter's tenure (new starter still starting, season_prior_starts_team<4)
print("--- Extension: weeks 1-4, team's week-1 new starter still starting (<=4th start with team this season)")
for s in ["home","away"]:
    m[f"{s}_new4"]=((m[f"{s}_new"]==1)&(m[f"{s}_season_prior_starts_team"]<4)&(m[f"{s}_stint_idx"]==m[f"{s}_season_prior_starts_team"]+1)&(m.week<=4)).astype(int)
    m[f"{s}_first4"]=((m[f"{s}_new4"]==1)&(m[f"{s}_career_prior_starts"]<4)).astype(int)
m["net_new4"]=m.home_new4-m.away_new4; m["net_first4"]=m.home_first4-m.away_first4
e=m[(m.game_type=="REG")&(m.week<=4)]
for f,lab in [("new4","new starter (wk1-4)"),("first4","rookie-proxy new starter (wk1-4, <4 career starts)")]:
    print(f"  [{lab}]")
    ols("resid_mkt",f"net_{f}",e,"resid vs MARKET close"); ols("mkt_minus_noqb",f"net_{f}",e,"market move vs no-QB line"); ols("resid_noqb",f"net_{f}",e,"resid vs no-QB line"); ols("resid_base",f"net_{f}",e,"resid vs 538-adj line"); ats_fade(e,f"net_{f}","fade ATS")
print("--- Extension: ALL weeks, inexperienced starter (<10 career starts) — does the market misprice inexperience?")
ols("resid_mkt","net_inexp",m,"resid vs MARKET close (all)"); ols("resid_base","net_inexp",m,"resid vs 538-adj line (all)"); ols("mkt_minus_noqb","net_inexp",m,"market move vs no-QB line"); ats_fade(m,"net_inexp","fade inexperienced ATS")
ols("resid_mkt","net_inexp",m[m.season<=2021],"resid vs market <=2021"); ols("resid_mkt","net_inexp",m[m.season>=2022],"resid vs market 2022-2025")
