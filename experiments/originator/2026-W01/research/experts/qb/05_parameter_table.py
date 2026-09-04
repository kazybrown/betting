"""05: final parameter table — points value of a backup (D3 prospective downgrade stint) by stint x era:
market-implied (market close - no-QB nfelo line), realized (vs no-QB line), nfelo 538 adj assigned, residual after 538 adj,
residual vs market. Means with HC1 SE from OLS on backup_net. Also: nfelo's 538 adj scale and implied top-up."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.line_nfelo_noqb.notna()&m.mkt_spread.notna()].copy()
def net_stint(df,lo,hi):
    h=((df.home_down==1)&df.home_stint3.between(lo,hi)).astype(int); a=((df.away_down==1)&df.away_stint3.between(lo,hi)).astype(int); return h-a
def coef(y,x,d):
    d=d.dropna(subset=[y,x]); r=sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], int((d[x]!=0).sum())
rows=[]
for era,d in [("ALL",m),("2009-2015",m[m.era=="2009-2015"]),("2016-2025",m[m.era=="2016-2025"]),("2022-2025 (OOS window)",m[m.season>=2022])]:
    for st,(lo,hi) in [("1st start",(1,1)),("2nd-3rd",(2,3)),("4th+",(4,99)),("all stints",(1,99))]:
        dd=d.copy(); dd["x"]=net_stint(dd,lo,hi); dd=dd[(dd.x!=0)|(dd.net_D3==0)]
        out={"era":era,"stint":st}
        for y,name in [("mkt_minus_noqb","market_implied"),("resid_noqb","realized_vs_noQB"),("qb_adj_pts","nfelo538_adj"),("resid_base","resid_after_538"),("resid_mkt","resid_vs_market")]:
            b,se,n=coef(y,"x",dd); out[name]=f"{-b:+.2f}±{se:.2f}" if name!="market_implied" else f"{b:+.2f}±{se:.2f}"; out["n_event"]=n
        rows.append(out)
t=pd.DataFrame(rows)
print("Penalty charged to the team with the backup (points, + = backup team penalised); ±HC1 SE; realized/538/resid signs flipped so + = penalty")
print(t.to_string(index=False))
print("\nInterpretation columns: market_implied = how much the closing market penalises the backup relative to the no-QB nfelo rating line;")
print("realized_vs_noQB = points the backup team actually lost vs that line; nfelo538_adj = penalty nfelo's 538-style QB adj applied;")
print("resid_after_538 = penalty still unexplained after nfelo's adj (what a top-up would need to be); resid_vs_market = unexplained after the market.")
# implied top-up rule for nfelo unregressed line, fitted on <=2021 only
tr=m[m.season<=2021].copy(); tr["s1"]=net_stint(tr,1,1); tr["s23"]=net_stint(tr,2,3); tr["s4"]=net_stint(tr,4,99)
r=sm.OLS(tr.resid_base, sm.add_constant(tr[["s1","s23","s4"]])).fit(cov_type="HC1")
print("\nTop-up on nfelo UNREGRESSED 538-adj line, fitted <=2021 (pts): 1st=%.2f (se %.2f), 2nd-3rd=%.2f (se %.2f), 4th+=%.2f (se %.2f)" % (-r.params.s1,r.bse.s1,-r.params.s23,r.bse.s23,-r.params.s4,r.bse.s4))
r2=sm.OLS(tr.resid_noqb, sm.add_constant(tr[["s1","s23","s4"]])).fit(cov_type="HC1")
print("Full penalty on a NO-QB rating line, fitted <=2021 (pts): 1st=%.2f (se %.2f), 2nd-3rd=%.2f (se %.2f), 4th+=%.2f (se %.2f)" % (-r2.params.s1,r2.bse.s1,-r2.params.s23,r2.bse.s23,-r2.params.s4,r2.bse.s4))
r3=sm.OLS(tr.mkt_minus_noqb, sm.add_constant(tr[["s1","s23","s4"]])).fit(cov_type="HC1")
print("Market-implied penalty, fitted <=2021 (pts): 1st=%.2f (se %.2f), 2nd-3rd=%.2f (se %.2f), 4th+=%.2f (se %.2f)" % (r3.params.s1,r3.bse.s1,r3.params.s23,r3.bse.s23,r3.params.s4,r3.bse.s4))
# how much does the 538 adj scale with QB quality gap? distribution of 538 penalty at 1st start
d=m.copy(); d["s1"]=net_stint(d,1,1); ev=d[d.s1!=0]; pen=-(ev.qb_adj_pts*ev.s1)
print("\nDistribution of nfelo 538 penalty at 1st backup start (pts): p10=%.2f p25=%.2f median=%.2f p75=%.2f p90=%.2f  (n=%d)" % tuple(list(pen.quantile([.1,.25,.5,.75,.9]))+[len(pen)]))
mk=(ev.mkt_minus_noqb*ev.s1)
print("Distribution of MARKET-implied penalty at 1st backup start: p10=%.2f p25=%.2f median=%.2f p75=%.2f p90=%.2f" % tuple(mk.quantile([.1,.25,.5,.75,.9])))
print("corr(538 penalty, market-implied penalty) at 1st start: %.3f ; slope of market on 538: %.3f" % (np.corrcoef(pen,mk)[0,1], np.polyfit(pen,mk,1)[0]))
# Does the 538 penalty size predict the realized penalty beyond a flat value? (in-sample, descriptive)
rz=(-(ev.resid_noqb*ev.s1)).values
r4=sm.OLS(rz, sm.add_constant(pd.DataFrame({"pen538":pen.values}))).fit(cov_type="HC1")
print("realized penalty ~ 538 penalty at 1st start: intercept=%.2f slope=%.2f (se %.2f, p=%.3f)" % (r4.params.const, r4.params.pen538, r4.bse.pen538, r4.pvalues.pen538))
r5=sm.OLS(rz, sm.add_constant(pd.DataFrame({"mkt":mk.values}))).fit(cov_type="HC1")
print("realized penalty ~ market-implied penalty at 1st start: intercept=%.2f slope=%.2f (se %.2f, p=%.3f)" % (r5.params.const, r5.params.mkt, r5.bse.mkt, r5.pvalues.mkt))
