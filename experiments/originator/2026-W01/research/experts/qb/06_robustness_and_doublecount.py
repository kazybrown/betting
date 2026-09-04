"""06: (i) reconcile stint-1 event counts (02b vs 05 sample construction); (ii) stint-1 sensitivity to dropping
week>=17 + playoffs (rest games); (iii) Theory 4: confirm no non-QB injury fields exist locally; (iv) double-count
arithmetic for ORIGINATOR's blend using measured embedded nfelo QB penalties."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.line_nfelo_noqb.notna()&m.mkt_spread.notna()].copy()
h1=(m.home_down==1)&(m.home_stint3==1); a1=(m.away_down==1)&(m.away_stint3==1)
both=(m.home_down==1)&(m.away_down==1)
print("(i) games with a stint-1 D3 backup:", int((h1|a1).sum()), "| of which opponent also D3 backup (any stint):", int(((h1|a1)&both).sum()),
      "| both stint-1:", int((h1&a1).sum()))
print("    02b sample (drops opponent-also-backup games):", int(((h1|a1)&~both).sum()), " 05 sample (x=h1-a1 !=0):", int(((h1.astype(int)-a1.astype(int))!=0).sum()))
def coef(y,x,d):
    d=d.dropna(subset=[y,x]); r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], int((d[x]!=0).sum())
m["x1"]=h1.astype(int)-a1.astype(int)
print("(ii) stint-1 D3, clean sample (opponent not a backup) vs excluding wk17+/playoffs:")
for lab,d in [("clean, all weeks", m[(m.x1!=0)&~both | (m.net_D3==0)]),
              ("clean, REG wk<=16 only", m[((m.x1!=0)&~both | (m.net_D3==0)) & (m.game_type=="REG") & (m.week<=16)]),
              ("clean, 2016-2025, REG wk<=16", m[((m.x1!=0)&~both | (m.net_D3==0)) & (m.game_type=="REG") & (m.week<=16) & (m.season>=2016)])]:
    out=[]
    for y,nm,sg in [("mkt_minus_noqb","market",1),("resid_noqb","realized",-1),("qb_adj_pts","538adj",-1),("resid_base","after538",-1),("resid_mkt","vs_mkt",-1)]:
        b,se,n=coef(y,"x1",d); out.append(f"{nm}={sg*b:+.2f}±{se:.2f}")
    print(f"    {lab:<32s} n_event={n:3d}  "+"  ".join(out))
# (iii) injury fields
g=pd.read_csv("/home/user/originator-2026-w01/research/data/games_1999_2025.csv", nrows=2)
inj=[c for c in g.columns if any(k in c.lower() for k in ["inj","status","active","out","questionable"])]
print("(iii) games.csv columns matching injury keywords:", inj if inj else "NONE — only QB names/ids (home_qb_name, away_qb_name) identify personnel")
others=[p for p in Path("/home/user/originator-2026-w01/research/data").glob("*.csv")]
hits=[]
for p in others:
    cols=pd.read_csv(p, nrows=1).columns
    hits+= [f"{p.name}:{c}" for c in cols if any(k in c.lower() for k in ["inj","wr","ol_","oline","pass_rush","status"])]
print("     other files with WR/OL/injury-like columns:", hits if hits else "NONE")
# (iv) double-count arithmetic
print("(iv) double-count arithmetic for ORIGINATOR = 0.46*nfelo + 0.39*PFF + 0.15*Cole")
tr=m[m.season<=2021]
emb={}
for st,(lo,hi) in [("1st",(1,1)),("2nd-3rd",(2,3)),("4th+",(4,99))]:
    x=((tr.home_down==1)&tr.home_stint3.between(lo,hi)).astype(int)-((tr.away_down==1)&tr.away_stint3.between(lo,hi)).astype(int)
    d=tr.assign(x=x); d=d[(d.x!=0)|(d.net_D3==0)]
    b_m,_,_=coef("mkt_minus_noqb","x",d); b_5,_,_=coef("qb_adj_pts","x",d); emb[st]=(b_m,-b_5)
    print(f"    {st:<8s} market-implied target P={b_m:.2f}  nfelo embeds A_nfelo={-b_5:.2f}  -> nfelo-share top-up 0.46*(P-A)={0.46*(b_m+b_5):+.2f}")
print("    If ORIGINATOR adds the spec's -2.0..-4.5 on top of the blend while nfelo already embeds A_nfelo (and PFF/Cole are QB-adjusted, which is")
print("    the norm for weekly power ratings), effective penalty at a 1st start = 0.46*A_nfelo + 0.39*P_pff + 0.15*P_cole + spec_adj; with all three")
P,A=emb["1st"]
for spec in [2.0,3.0,4.5]:
    print(f"      inputs QB-adjusted at ~P={P:.2f}: {P:.2f}+{spec:.1f} = {P+spec:.2f} vs market {P:.2f} -> overshoot {spec:.1f} pts; with only nfelo adjusted: {0.46*A+spec:.2f} vs target {P:.2f} -> {'over' if 0.46*A+spec>P else 'under'}shoot {abs(0.46*A+spec-P):.2f}")
