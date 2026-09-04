"""07: separate in-season backup starts (REG weeks<=16) from week 17+/playoff 'rest' backup starts; re-run the OOS
stint-1 top-up test on the clean in-season sample; era split of the clean stint-1 numbers."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import ats
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.line_nfelo_noqb.notna()&m.mkt_spread.notna()&m.nfelo_home_line_close.notna()].copy()
both=(m.home_down==1)&(m.away_down==1)
def ns(df,lo,hi): return ((df.home_down==1)&df.home_stint3.between(lo,hi)).astype(int)-((df.away_down==1)&df.away_stint3.between(lo,hi)).astype(int)
m["s1"]=ns(m,1,1); m["s23"]=ns(m,2,3); m["s4"]=ns(m,4,99); m["resid_close"]=m.margin+m.nfelo_home_line_close
m["late"]=((m.game_type!="REG")|(m.week>=17)).astype(int)
def coef(y,x,d):
    d=d.dropna(subset=[y,x]); r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], int((d[x]!=0).sum())
def row(lab,d,x="s1"):
    out=[]
    for y,nm,sg in [("mkt_minus_noqb","market",1),("resid_noqb","realized",-1),("qb_adj_pts","538adj",-1),("resid_base","after538",-1),("resid_mkt","vs_mkt",-1)]:
        b,se,n=coef(y,x,d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}")
    print(f"  {lab:<44s} n_event={n:3d}  "+"  ".join(out))
clean=m[~both]
print("A) stint-1 D3 backup, opponent not a backup — in-season (REG wk<=16) vs late (wk17+/playoffs) vs era")
row("REG wk<=16, ALL", clean[(clean.late==0)&((clean.s1!=0)|(clean.net_D3==0))])
row("REG wk<=16, 2009-2015", clean[(clean.late==0)&(clean.season<=2015)&((clean.s1!=0)|(clean.net_D3==0))])
row("REG wk<=16, 2016-2025", clean[(clean.late==0)&(clean.season>=2016)&((clean.s1!=0)|(clean.net_D3==0))])
row("REG wk<=16, 2022-2025", clean[(clean.late==0)&(clean.season>=2022)&((clean.s1!=0)|(clean.net_D3==0))])
row("wk17+ / playoffs (rest games), ALL", clean[(clean.late==1)&((clean.s1!=0)|(clean.net_D3==0))])
print("  stint 2-3, REG wk<=16:"); row("REG wk<=16, ALL, stint 2-3", clean[(clean.late==0)&((clean.s23!=0)|(clean.net_D3==0))], x="s23")
print("  stint 4+, REG wk<=16:"); row("REG wk<=16, ALL, stint 4+", clean[(clean.late==0)&((clean.s4!=0)|(clean.net_D3==0))], x="s4")
print("  ANY D3 stint, REG wk<=16:"); row("REG wk<=16, ALL, any stint", clean[(clean.late==0)], x="net_D3")

print("\nB) OOS (fit<=2021, test 2022-2025) restricted to REG weeks<=16: penalties on nfelo 538-adj (unregressed) line, nfelo close, market")
rng=np.random.default_rng(11)
def boot(e1,e0,B=4000):
    d=np.abs(np.asarray(e1))-np.abs(np.asarray(e0)); bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(B)]); return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
ins=m[m.late==0]; tr=ins[ins.season<=2021]; te=ins[ins.season>=2022]
print(f"  train n={len(tr)} test n={len(te)}; test stint-1 events={int((te.s1!=0).sum())}, any-D3 events={int((te.net_D3!=0).sum())}")
for y,basecol,lab in [("resid_base","line_nfelo_base","nfelo 538-adj line"),("resid_close","nfelo_home_line_close","nfelo close"),("resid_mkt","mkt_spread","market close"),("resid_noqb","line_nfelo_noqb","no-QB line")]:
    k1=-coef(y,"s1",tr)[0]; kall=-coef(y,"net_D3",tr)[0]
    base=te[basecol]; e0=te.margin+base
    for nm,adj in [(f"stint-1 penalty {k1:.2f}", base+k1*te.s1),(f"all-stint flat penalty {kall:.2f}", base+kall*te.net_D3),("stint-1 penalty 1.0", base+1.0*te.s1),("stint-1 penalty 2.0", base+2.0*te.s1)]:
        e1=te.margin+adj; dm,lo,hi=boot(e1,e0); w,l,p=ats(adj,te.mkt_spread,te.margin)
        ev=te[te.s1!=0] if "stint-1" in nm else te[te.net_D3!=0]
        de=(np.abs(ev.margin+adj.loc[ev.index]).mean()-np.abs(ev.margin+ev[basecol]).mean())
        print(f"   {lab:<20s} + {nm:<28s} MAE {np.abs(e0).mean():.4f} -> {np.abs(e1).mean():.4f}  dMAE={dm:+.4f} [{lo:+.4f},{hi:+.4f}]  on event games dMAE={de:+.3f} (n={len(ev)})  ATS vs mkt {w}-{l}-{p} ({w/max(1,w+l):.3f})")
print("\nC) fade-the-backup ATS vs close, stint 1, REG wk<=16, 2022-2025 only (pure OOS):")
from scipy import stats
d=te[te.s1!=0]; win=((d.s1==1)&(d.resid_mkt<0))|((d.s1==-1)&(d.resid_mkt>0)); push=d.resid_mkt==0; w=int(win.sum()); l=int((~win&~push).sum())
ci=stats.binomtest(w,w+l).proportion_ci(0.95); print(f"   {w}-{l}-{int(push.sum())} = {w/(w+l):.3f} 95%CI=[{ci.low:.3f},{ci.high:.3f}] p={stats.binomtest(w,w+l,0.5).pvalue:.3f}")
