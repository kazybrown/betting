"""THEORY 5 (cont): divisional-game totals vs market, fit vs test windows, and nfelo's div mod bias."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from kit import merged
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].dropna(subset=["mkt_total","mkt_spread"]).copy()
m["div"] = m.div_game.astype(int)
for lab, d in [("2009-2013", m[(m.season>=2009)&(m.season<=2013)]), ("2014-2019", m[(m.season>=2014)&(m.season<=2019)]),
               ("FIT 2009-2021 ex2020", m[(m.season<=2021)&(m.season!=2020)]), ("TEST 2022-2025", m[m.season>=2022])]:
    a, b = d[d["div"]==1], d[d["div"]==0]
    tt = stats.ttest_ind(a.total_err_mkt, b.total_err_mkt, equal_var=False)
    print("%-22s div: n=%4d total=%.2f mkt=%.2f resid=%+.2f (se %.2f) over=%.3f | non-div: n=%4d total=%.2f mkt=%.2f resid=%+.2f (se %.2f) over=%.3f | diff=%+.2f p=%.3f" %
          (lab, len(a), a.total_pts.mean(), a.mkt_total.mean(), a.total_err_mkt.mean(), a.total_err_mkt.std()/np.sqrt(len(a)), (a.total_err_mkt>0).sum()/(a.total_err_mkt!=0).sum(),
           len(b), b.total_pts.mean(), b.mkt_total.mean(), b.total_err_mkt.mean(), b.total_err_mkt.std()/np.sqrt(len(b)), (b.total_err_mkt>0).sum()/(b.total_err_mkt!=0).sum(),
           a.total_err_mkt.mean()-b.total_err_mkt.mean(), tt.pvalue))
# with season FE
fit = m[(m.season<=2021)&(m.season!=2020)]; test = m[m.season>=2022]
mf = smf.ols("total_err_mkt ~ div + C(season)", data=fit).fit(cov_type="HC1"); mt = smf.ols("total_err_mkt ~ div + C(season)", data=test).fit(cov_type="HC1")
print("market total residual ~ div + season FE: FIT div=%+.2f (se %.2f, p=%.3f) | TEST div=%+.2f (se %.2f, p=%.3f)" %
      (mf.params["div"], mf.bse["div"], mf.pvalues["div"], mt.params["div"], mt.bse["div"], mt.pvalues["div"]))
# OOS: shifting the market total by fit-window div coefficient -- does it reduce test MAE / improve O/U?
adj = mf.params["div"]
e0 = test.total_pts - test.mkt_total; e1 = e0 - adj*test["div"]
print("OOS totals MAE: market=%.3f | market + %.2f*div=%.3f ; div-game under rate OOS=%.3f (n=%d)" %
      (e0.abs().mean(), adj, e1.abs().mean(), (e0[test["div"]==1]<0).sum()/(e0[test["div"]==1]!=0).sum(), (test["div"]==1).sum()))
# nfelo div mod: bias of nfelo close line in div vs non-div, fit and test
for lab, d in [("FIT", fit), ("TEST", test)]:
    d = d.dropna(subset=["nfelo_home_line_close"])
    for dv in [0,1]:
        x = d[d["div"]==dv]; e = x.margin + x.nfelo_home_line_close
        print("  nfelo close line %-4s div=%d n=%4d bias=%+.2f (se %.2f) MAE=%.3f | market bias=%+.2f MAE=%.3f" % (lab, dv, len(x), e.mean(), e.std()/np.sqrt(len(x)), e.abs().mean(), x.spread_err_mkt.mean(), x.spread_err_mkt.abs().mean()))
