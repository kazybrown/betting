"""THEORY 2 (cont): (a) does ORIGINATOR's site HFA = (hfa_mod + home_time_advantage_mod)/25 double count the
time-zone term?  (b) Denver (altitude) as the one venue with a physical mechanism and the highest market HFA."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from kit import merged
pd.set_option("display.width", 220)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy()
m["hfa_resid"] = m.margin - m.rating_dif; m["mkt_hfa"] = m.spread_line - m.rating_dif
m["hfa_base"] = m.hfa_mod/25; m["tz"] = m.home_time_advantage_mod.fillna(0)/25
m["hfa_orig"] = m.hfa_pts   # what ORIGINATOR's PFF/Cole path uses
print("(a) time-zone term. tz (pts) value counts 2022-25:", (m[m.season>=2022].tz.round(2)).value_counts().head(8).to_dict())
print("    corr(hfa_base, tz) 2022-25 = %.2f ; mean hfa_base by tz sign:" % np.corrcoef(m[m.season>=2022].hfa_base, m[m.season>=2022].tz)[0,1],
      m[m.season>=2022].groupby(np.sign(m[m.season>=2022].tz)).hfa_base.mean().round(2).to_dict())
for lo, hi in [(2009,2019),(2021,2025),(2022,2025)]:
    d = m[(m.season>=lo)&(m.season<=hi)]
    o = sm.OLS(d.hfa_resid, sm.add_constant(d[["hfa_base","tz"]])).fit(cov_type="HC1")
    om = sm.OLS(d.mkt_hfa, sm.add_constant(d[["hfa_base","tz"]])).fit(cov_type="HC1")
    print("    %d-%d realized resid ~ hfa_base + tz: b_base=%.2f (se %.2f) b_tz=%.2f (se %.2f) | market-implied HFA ~ same: b_base=%.2f b_tz=%.2f" %
          (lo, hi, o.params["hfa_base"], o.bse["hfa_base"], o.params["tz"], o.bse["tz"], om.params["hfa_base"], om.params["tz"]))
d = m[m.season>=2021]
print("    2021-25 realized HFA by tz bucket:")
for b, x in d.groupby(pd.cut(d.tz, [-2, -0.2, 0.2, 2], labels=["home disadv (tz<-0.2)","~0","home adv (tz>0.2)"]), observed=True):
    print("      %-24s n=%4d realized=%.2f (se %.2f) mkt_impl=%.2f nfelo hfa_orig=%.2f nfelo hfa_base=%.2f" % (b, len(x), x.hfa_resid.mean(), x.hfa_resid.std()/np.sqrt(len(x)), x.mkt_hfa.mean(), x.hfa_orig.mean(), x.hfa_base.mean()))
# spread of ORIGINATOR's per-game site HFA vs a constant
print("    SD of nfelo per-game site HFA used by ORIGINATOR (2022-25): %.2f pts; range %.2f..%.2f" % (d[d.season>=2022].hfa_orig.std(), d[d.season>=2022].hfa_orig.min(), d[d.season>=2022].hfa_orig.max()))

print("\n(b) Denver")
lg = m.groupby("season").hfa_resid.transform("mean"); m["excess"] = m.hfa_resid - lg
lgm = m.groupby("season").mkt_hfa.transform("mean"); m["mkt_excess"] = m.mkt_hfa - lgm
for lab, d in [("2009-2021 ex2020", m[(m.season<=2021)&(m.season!=2020)]), ("2022-2025", m[m.season>=2022]), ("2009-2025 ex2020", m[m.season!=2020])]:
    x = d[d.home.eq("DEN")]
    tt = stats.ttest_1samp(x.excess, 0)
    print("  %-17s n=%3d realized excess=%+.2f (se %.2f, p=%.3f) | market-implied excess=%+.2f | nfelo hfa_orig=%.2f (league %.2f)" %
          (lab, len(x), x.excess.mean(), x.excess.std()/np.sqrt(len(x)), tt.pvalue, x.mkt_excess.mean(), x.hfa_orig.mean(), d.hfa_orig.mean()))
# OOS Denver-only bump on top of constant HFA
fit = m[(m.season<=2021)&(m.season!=2020)]; test = m[m.season>=2022]; K = fit.hfa_resid.mean()
den = test[test.home.eq("DEN")]
for bump in [0, 0.5, 1.0, 1.5]:
    e = den.margin - den.rating_dif - K - bump
    print("  OOS DEN home games (n=%d) const K + %.1f: MAE=%.2f bias=%+.2f" % (len(den), bump, e.abs().mean(), e.mean()))
print("  OOS DEN market: MAE=%.2f bias=%+.2f home cover %d-%d" % (den.spread_err_mkt.abs().mean(), den.spread_err_mkt.mean(), (den.spread_err_mkt>0).sum(), (den.spread_err_mkt<0).sum()))
# Same for the other named venues: full-sample excess with p-values and market excess
print("\n  Named venues, 2009-2025 ex 2020 (realized excess over league, market-implied excess):")
for t in ["SEA","DEN","KC","GB","NO","BAL","PIT","NE","MIN","BUF","ARI","LAC","LA","LV","JAX","TB","NYG","WAS","CLE"]:
    x = m[(m.home.eq(t))&(m.season!=2020)]; x2 = x[x.season>=2022]
    print("   %-4s n=%3d realized=%+.2f (se %.2f, p=%.2f) market=%+.2f | 2022-25: realized=%+.2f market=%+.2f nfelo_excess=%+.2f" %
          (t, len(x), x.excess.mean(), x.excess.std()/np.sqrt(len(x)), stats.ttest_1samp(x.excess,0).pvalue, x.mkt_excess.mean(),
           x2.excess.mean(), x2.mkt_excess.mean(), x2.hfa_orig.mean()-test.hfa_orig.mean()))
