"""03: OUT-OF-SAMPLE tests (fit <=2021, test 2022-2025; plus rolling-origin by season).
Adjustment convention: penalty p>0 points charged to the team with the backup:  line_new = line + p*backup_net
(line in ORIGINATOR convention, negative = home favored; backup_net=+1 home backup). p = -(fitted residual coef).
T1: flat / stint-specific penalty on a no-QB rating line vs nfelo's 538 adj (and rescaled 538 adj).
T2: extra penalty on top of the MARKET close. T3: extra penalty on top of nfelo's regressed close.
Paired bootstrap CI on MAE differences (negative dMAE = adjustment helps)."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import mae, ats
HERE=Path(__file__).resolve().parent
m=pd.read_csv(HERE/"qb_games_defs.csv", low_memory=False)
m=m[m.line_nfelo_noqb.notna()&m.mkt_spread.notna()&m.nfelo_home_line_close.notna()].copy()
m["resid_close"]=m.margin+m.nfelo_home_line_close
# stint-specific nets for D3: stint 1, stint 2-3, stint 4+
def net_stint(df,lo,hi):
    h=((df.home_down==1)&df.home_stint3.between(lo,hi)).astype(int); a=((df.away_down==1)&df.away_stint3.between(lo,hi)).astype(int); return h-a
m["net_D3_s1"]=net_stint(m,1,1); m["net_D3_s23"]=net_stint(m,2,3); m["net_D3_s4"]=net_stint(m,4,99)
rng=np.random.default_rng(7)
def boot_diff(e1,e2,B=4000):
    e1,e2=np.asarray(e1),np.asarray(e2); d=np.abs(e1)-np.abs(e2); n=len(d)
    bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)]); return d.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
def fit(train,y,xs):
    d=train.dropna(subset=[y]+xs); r=sm.OLS(d[y], sm.add_constant(d[xs])).fit(); return r.params[xs].to_dict()
def report(name, pred, test, base_pred):
    e=test.margin+pred; eb=test.margin+base_pred; dm,lo,hi=boot_diff(e,eb); w,l,p=ats(pred, test.mkt_spread, test.margin)
    print(f"   {name:<58s} MAE={mae(-pred,test.margin):.4f} RMSE={np.sqrt(np.mean(e**2)):.4f} dMAE={dm:+.4f} [{lo:+.4f},{hi:+.4f}] ATS {w}-{l}-{p} ({w/max(1,w+l):.3f})")

train=m[m.season<=2021]; test=m[m.season>=2022]
print(f"train n={len(train)} (2009-2021), test n={len(test)} (2022-2025)")
print("test event games: D1=%d D3=%d D3 stint1=%d stint2-3=%d stint4+=%d" % tuple(int((test[c]!=0).sum()) for c in ["net_D1","net_D3","net_D3_s1","net_D3_s23","net_D3_s4"]))
for defn in ["net_D3","net_D1"]:
    print(f"\n######## definition {defn} ########")
    p_noqb=-fit(train,"resid_noqb",[defn])[defn]; p_base=-fit(train,"resid_base",[defn])[defn]; p_mkt=-fit(train,"resid_mkt",[defn])[defn]; p_close=-fit(train,"resid_close",[defn])[defn]
    s538=fit(train,"resid_noqb",["qb_adj_pts"])["qb_adj_pts"]
    print(f" fitted penalties on <=2021 (pts charged to backup team): on no-QB line={p_noqb:.2f}  on 538-adj line={p_base:.2f}  on market={p_mkt:.2f}  on nfelo close={p_close:.2f} | 538 scale s={s538:.3f}")
    print(" T1: rating line WITHOUT QB adj as base")
    base=test.line_nfelo_noqb
    report("no-QB line (base)", base, test, base)
    report(f"no-QB + flat penalty {p_noqb:.2f} (fitted)", base + p_noqb*test[defn], test, base)
    for kk in [1.0,2.0,3.0,4.0]: report(f"no-QB + flat penalty {kk:.1f}", base + kk*test[defn], test, base)
    report("no-QB + nfelo 538 adj (= line_nfelo_base)", test.line_nfelo_base, test, base)
    report(f"no-QB + {s538:.2f} x 538 adj (rescaled, fitted)", base - s538*test.qb_adj_pts, test, base)
    report(f"538-adj line + flat penalty {p_base:.2f} (fitted)", test.line_nfelo_base + p_base*test[defn], test, base)
    print(" T2: MARKET close as base")
    base=test.mkt_spread
    report("market close (base)", base, test, base)
    report(f"market + penalty {p_mkt:.2f} (fitted)", base + p_mkt*test[defn], test, base)
    for kk in [0.5,1.0,2.0]: report(f"market + penalty {kk:.1f}", base + kk*test[defn], test, base)
    print(" T3: nfelo regressed close as base")
    base=test.nfelo_home_line_close
    report("nfelo close (base)", base, test, base)
    report(f"nfelo close + penalty {p_close:.2f} (fitted)", base + p_close*test[defn], test, base)
    for kk in [0.5,1.0,2.0]: report(f"nfelo close + penalty {kk:.1f}", base + kk*test[defn], test, base)

print("\n######## stint-specific penalties (D3) fitted on <=2021, tested 2022-2025 ########")
xs=["net_D3_s1","net_D3_s23","net_D3_s4"]
for y,basecol,lab in [("resid_noqb","line_nfelo_noqb","no-QB line"),("resid_base","line_nfelo_base","538-adj line"),("resid_mkt","mkt_spread","market close"),("resid_close","nfelo_home_line_close","nfelo close")]:
    k=fit(train,y,xs); pen={x:-k[x] for x in xs}
    print(f" base={lab:<13s} fitted penalties: stint1={pen['net_D3_s1']:.2f} stint2-3={pen['net_D3_s23']:.2f} stint4+={pen['net_D3_s4']:.2f}")
    base=test[basecol]; adj=base+sum(pen[x]*test[x] for x in xs)
    report(f"   {lab} + stint penalties (fitted)", adj, test, base)
    adj2=base+pen['net_D3_s1']*test.net_D3_s1
    report(f"   {lab} + stint-1 penalty only", adj2, test, base)
    # evaluate only on event games for power
    ev=test[(test.net_D3!=0)]
    e0=ev.margin+ev[basecol]; e1=ev.margin+ev[basecol]+sum(pen[x]*ev[x] for x in xs); dm,lo,hi=boot_diff(e1,e0)
    print(f"      on D3 event games only (n={len(ev)}): MAE base={np.abs(e0).mean():.3f} adj={np.abs(e1).mean():.3f} dMAE={dm:+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n######## rolling-origin (fit on all prior seasons, test one season) — D3 ########")
for yr in range(2018,2026):
    tr=m[m.season<yr]; te=m[m.season==yr]
    p1=-fit(tr,"resid_noqb",["net_D3"])["net_D3"]; p2=-fit(tr,"resid_mkt",["net_D3"])["net_D3"]; s=fit(tr,"resid_noqb",["qb_adj_pts"])["qb_adj_pts"]
    k=fit(tr,"resid_close",xs); pen={x:-k[x] for x in xs}
    e_noqb=te.margin+te.line_nfelo_noqb; e_k=e_noqb+p1*te.net_D3; e_538=te.margin+te.line_nfelo_base
    e_m=te.margin+te.mkt_spread; e_mk=e_m+p2*te.net_D3; e_c=te.margin+te.nfelo_home_line_close; e_cs=e_c+sum(pen[x]*te[x] for x in xs)
    print(f"  {yr}: p_noqb={p1:.2f} p_mkt={p2:.2f} s538={s:.2f} stint-pen(close)={pen['net_D3_s1']:.2f}/{pen['net_D3_s23']:.2f}/{pen['net_D3_s4']:.2f} | MAE noQB={np.abs(e_noqb).mean():.3f} +flat={np.abs(e_k).mean():.3f} 538={np.abs(e_538).mean():.3f} | mkt={np.abs(e_m).mean():.4f} mkt+p={np.abs(e_mk).mean():.4f} | close={np.abs(e_c).mean():.4f} close+stint={np.abs(e_cs).mean():.4f} | ev={int((te.net_D3!=0).sum())}")
print("\n######## 2022-2025 test-window effect sizes (descriptive, HC1) ########")
for defn in ["net_D3","net_D3_s1","net_D1"]:
    for y in ["mkt_minus_noqb","resid_noqb","resid_base","resid_mkt","qb_adj_pts"]:
        d=test.dropna(subset=[y]); r=sm.OLS(d[y], sm.add_constant(d[[defn]])).fit(cov_type="HC1")
        print(f"  {defn:<10s} {y:<15s} coef={r.params[defn]:+.3f} se={r.bse[defn]:.3f} p={r.pvalues[defn]:.4f} n_event={int((d[defn]!=0).sum())}")
