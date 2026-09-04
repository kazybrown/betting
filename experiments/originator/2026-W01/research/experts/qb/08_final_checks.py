"""08: (a) clean in-season stint-1 numbers on the FIT window (2009-2021) vs TEST window (2022-2025) — is nfelo's
under-pricing of first backup starts a training-period fact or only a 2022-25 fact?  (b) fit/test split of the
week-1..4 'rookie-proxy new starter' market-beating effect from 04 (multiple-comparison guard)."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.line_nfelo_noqb.notna()&m.mkt_spread.notna()].copy()
both=(m.home_down==1)&(m.away_down==1)
m["s1"]=((m.home_down==1)&(m.home_stint3==1)).astype(int)-((m.away_down==1)&(m.away_stint3==1)).astype(int)
m["late"]=((m.game_type!="REG")|(m.week>=17)).astype(int)
def coef(y,x,d):
    d=d.dropna(subset=[y,x]); r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], r.pvalues[x], int((d[x]!=0).sum())
def row(lab,d,x):
    out=[]
    for y,nm,sg in [("mkt_minus_noqb","market",1),("resid_noqb","realized",-1),("qb_adj_pts","538adj",-1),("resid_base","after538",-1),("resid_mkt","vs_mkt",-1)]:
        b,se,p,n=coef(y,x,d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}(p={p:.2f})")
    print(f"  {lab:<40s} n_event={n:3d}  "+"  ".join(out))
clean=m[~both & ((m.s1!=0)|(m.net_D3==0))]
print("(a) stint-1 D3 backup, in-season REG wk<=16, opponent not a backup")
row("FIT window 2009-2021", clean[(clean.late==0)&(clean.season<=2021)], "s1")
row("TEST window 2022-2025", clean[(clean.late==0)&(clean.season>=2022)], "s1")
row("FIT window, wk17+/playoffs", clean[(clean.late==1)&(clean.season<=2021)], "s1")
row("TEST window, wk17+/playoffs", clean[(clean.late==1)&(clean.season>=2022)], "s1")
# (b) rookie-proxy new starter weeks 1-4
for s in ["home","away"]:
    m[f"{s}_new"]=(m[f"{s}_qb"]!=m[f"{s}_prev_season_primary"]).astype(int)
    m[f"{s}_new4"]=((m[f"{s}_new"]==1)&(m[f"{s}_season_prior_starts_team"]<4)&(m[f"{s}_stint_idx"]==m[f"{s}_season_prior_starts_team"]+1)&(m.week<=4)).astype(int)
    m[f"{s}_first4"]=((m[f"{s}_new4"]==1)&(m[f"{s}_career_prior_starts"]<4)).astype(int)
m["net_first4"]=m.home_first4-m.away_first4; m["net_new4"]=m.home_new4-m.away_new4
e=m[(m.game_type=="REG")&(m.week<=4)]
print("(b) week 1-4 new starter with <4 career starts (rookie proxy): resid vs market, fit vs test")
for lab,d in [("2009-2021",e[e.season<=2021]),("2022-2025",e[e.season>=2022]),("2009-2015",e[e.season<=2015]),("2016-2025",e[e.season>=2016])]:
    b,se,p,n=coef("resid_mkt","net_first4",d)
    dd=d[d.net_first4!=0]; win=((dd.net_first4==1)&(dd.resid_mkt>0))|((dd.net_first4==-1)&(dd.resid_mkt<0)); push=dd.resid_mkt==0; w=int(win.sum()); l=int((~win&~push).sum())
    print(f"  {lab}: coef={b:+.2f}±{se:.2f} p={p:.3f} n_event={n}  | BACK-the-rookie ATS {w}-{l}-{int(push.sum())} = {w/max(1,w+l):.3f} p={stats.binomtest(w,w+l,0.5).pvalue:.3f}")
print("  all new starters wk1-4 (any experience):")
for lab,d in [("2009-2021",e[e.season<=2021]),("2022-2025",e[e.season>=2022])]:
    b,se,p,n=coef("resid_mkt","net_new4",d); print(f"  {lab}: coef={b:+.2f}±{se:.2f} p={p:.3f} n_event={n}")
print("  number of hypothesis tests run against the market in 04 (week-1 family): ~20 -> Bonferroni-adjusted alpha ~0.0025")
