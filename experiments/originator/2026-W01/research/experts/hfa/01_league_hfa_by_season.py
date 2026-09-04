"""THEORY 1: League HFA by season 2009-2025.
HFA measured three ways per season (REG season, location==Home):
  raw       = mean(margin)
  rating    = mean(margin - rating_dif)  where rating_dif = nfelo QB-adjusted Elo diff / 25 (no HFA/bye/div mods)
  mkt_impl  = intercept of spread_line ~ rating_dif  (what the market prices as HFA)
  mkt_resid = mean(margin + mkt_spread)  (market's home bias; 0 = market HFA correct)
Then: trend test, and OUT-OF-SAMPLE (2022-2025) test of which HFA constant minimises error of a
rating-based spread, vs nfelo's per-game hfa_pts.  Fit window <= 2021.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from kit import merged, mae
pd.set_option("display.width", 220)

m = merged()
m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m["rating_dif_noqb"] = m.elo_dif_pts
m = m.dropna(subset=["rating_dif", "mkt_spread"]).copy()
print("n home REG games w/ nfelo & market:", len(m))
# sanity
print("sanity corr(mkt_spread, margin) = %.3f (must be strongly negative)" % np.corrcoef(m.mkt_spread, m.margin)[0,1])
print("corr(rating_dif, margin) = %.3f ; corr(rating_dif_noqb, margin) = %.3f -> use QB-adjusted" %
      (np.corrcoef(m.rating_dif, m.margin)[0,1], np.corrcoef(m.rating_dif_noqb, m.margin)[0,1]))
m["hfa_resid"] = m.margin - m.rating_dif

rows = []
for s, d in m.groupby("season"):
    X = sm.add_constant(d.rating_dif)
    ols_res = sm.OLS(d.margin, X).fit()
    ols_mkt = sm.OLS(d.spread_line, X).fit()
    rows.append(dict(season=s, n=len(d),
                     raw=d.margin.mean(),
                     rating=d.hfa_resid.mean(), rating_se=d.hfa_resid.std()/np.sqrt(len(d)),
                     ols_icpt=ols_res.params["const"], ols_slope=ols_res.params["rating_dif"],
                     mkt_impl=ols_mkt.params["const"], mkt_mean_line=d.spread_line.mean(),
                     mkt_resid=d.spread_err_mkt.mean(), mkt_resid_se=d.spread_err_mkt.std()/np.sqrt(len(d)),
                     nfelo_hfa=d.hfa_pts.mean(), nfelo_hfa_mod_only=(d.hfa_mod/25).mean(),
                     home_cover=((d.spread_err_mkt>0).sum()/((d.spread_err_mkt!=0).sum()))))
T = pd.DataFrame(rows).set_index("season").round(2)
print("\nLeague HFA by season (points):"); print(T.to_string())

# multi-season blocks
def block(lo, hi, excl2020=True):
    d = m[(m.season>=lo)&(m.season<=hi)]
    if excl2020: d = d[d.season!=2020]
    r = d.hfa_resid; e = d.spread_err_mkt
    X = sm.add_constant(d.rating_dif)
    mk = sm.OLS(d.spread_line, X).fit().params["const"]
    return dict(window=f"{lo}-{hi}", n=len(d), raw=d.margin.mean(), rating_hfa=r.mean(), se=r.std()/np.sqrt(len(d)),
                ci_lo=r.mean()-1.96*r.std()/np.sqrt(len(d)), ci_hi=r.mean()+1.96*r.std()/np.sqrt(len(d)),
                mkt_impl=mk, mkt_resid=e.mean(), mkt_resid_se=e.std()/np.sqrt(len(d)), nfelo_hfa=d.hfa_pts.mean())
B = pd.DataFrame([block(2009,2013), block(2014,2019), block(2021,2025), block(2022,2025), block(2009,2021), block(2017,2021)])
print("\nBlocks (2020 excluded):"); print(B.round(2).to_string(index=False))

# trend test on season-level rating HFA (weighted by n), excluding 2020
t = T[T.index!=2020].reset_index()
w = sm.WLS(t.rating, sm.add_constant(t.season.astype(float)), weights=t.n).fit()
print("\nTrend (rating HFA ~ season, 2020 excl): slope=%.3f pts/yr, p=%.4f, 95%% CI [%.3f, %.3f]" %
      (w.params.iloc[1], w.pvalues.iloc[1], *w.conf_int().iloc[1]))
w2 = sm.WLS(t.mkt_impl, sm.add_constant(t.season.astype(float)), weights=t.n).fit()
print("Trend (market-implied HFA ~ season): slope=%.3f pts/yr, p=%.4f" % (w2.params.iloc[1], w2.pvalues.iloc[1]))
a = m[(m.season<=2019)].hfa_resid; b = m[(m.season>=2021)].hfa_resid
tt = stats.ttest_ind(a, b, equal_var=False)
print("Welch t-test 2009-2019 (mean %.2f, n=%d) vs 2021-2025 (mean %.2f, n=%d): diff=%.2f, t=%.2f, p=%.4f" %
      (a.mean(), len(a), b.mean(), len(b), a.mean()-b.mean(), tt.statistic, tt.pvalue))

# ---------------- OUT-OF-SAMPLE: which constant HFA for 2022-2025? -----------------
test = m[m.season>=2022].copy()
print("\n=== OUT-OF-SAMPLE 2022-2025 (n=%d): spread = -(rating_dif + k) ===" % len(test))
fits = {"k=fit 2009-2021 excl2020": m[(m.season<=2021)&(m.season!=2020)].hfa_resid.mean(),
        "k=fit 2017-2021 excl2020": m[(m.season>=2017)&(m.season<=2021)&(m.season!=2020)].hfa_resid.mean(),
        "k=fit 2021 only": m[m.season==2021].hfa_resid.mean()}
grid = {f"k={k:.1f}": k for k in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]}
res = []
def evalk(name, k):
    pred = -(test.rating_dif + k)
    err = test.margin + pred
    res.append(dict(spec=name, k=float(np.mean(k)) if np.ndim(k) else k, MAE=mae(pred, -test.margin) if False else np.abs(err).mean(),
                    RMSE=np.sqrt((err**2).mean()), bias=err.mean(), bias_se=err.std()/np.sqrt(len(err))))
for nm, k in {**fits, **grid}.items(): evalk(nm, k)
evalk("k=nfelo hfa_pts (per game)", test.hfa_pts.values)
evalk("k=nfelo hfa_mod/25 (per game)", (test.hfa_mod/25).values)
R = pd.DataFrame(res); print(R.round(3).to_string(index=False))
# rolling origin: k from prior 3 seasons (excl 2020) vs prior 10
print("\nRolling-origin: k = mean rating-HFA of prior W seasons (2020 excluded)")
for W in [1, 3, 5, 10]:
    errs = []
    for s in [2022, 2023, 2024, 2025]:
        tr = m[(m.season<s)&(m.season>=s-W-(1 if s-W<=2020 else 0))&(m.season!=2020)]
        k = tr.hfa_resid.mean(); d = test[test.season==s]
        errs.append(d.margin - (d.rating_dif + k))
    e = pd.concat(errs)
    print("  W=%2d: MAE=%.3f bias=%+.3f" % (W, e.abs().mean(), e.mean()))
# market benchmark on same games
e_m = test.spread_err_mkt
print("\nMarket close on same games: MAE=%.3f bias=%+.3f (se %.3f) home cover rate=%.3f" %
      (e_m.abs().mean(), e_m.mean(), e_m.std()/np.sqrt(len(e_m)), (e_m>0).sum()/(e_m!=0).sum()))
# market-implied HFA 2022-2025 by dome/outdoor (descriptive) and realized rating-HFA by roof
print("\n2021-2025 by roof (2020 excl): realized rating-HFA vs market-implied vs nfelo")
d5 = m[(m.season>=2021)]
for roof, d in d5.groupby(d5.is_dome.map({True:"dome/closed", False:"outdoor/open"})):
    X = sm.add_constant(d.rating_dif); mk = sm.OLS(d.spread_line, X).fit().params["const"]
    print("  %-13s n=%4d realized=%.2f (se %.2f) mkt_impl=%.2f mkt_resid=%+.2f nfelo_hfa=%.2f" %
          (roof, len(d), d.hfa_resid.mean(), d.hfa_resid.std()/np.sqrt(len(d)), mk, d.spread_err_mkt.mean(), d.hfa_pts.mean()))
# Does market residual predict anything? test whether home teams are over/under-valued 2022-2025 by season
print("\nMarket home bias by season 2021-2025 (mean spread_err_mkt, +=home under-priced):")
print(T.loc[2021:, ["n","mkt_resid","mkt_resid_se","home_cover"]].to_string())
