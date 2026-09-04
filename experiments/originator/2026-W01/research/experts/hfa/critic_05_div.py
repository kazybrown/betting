"""CRITIC HFA-5: (a) spread: rolling-origin pooled test of div-specific HFA; (b) totals: CONFOUNDS -- divisional games are
played later in the season (mean week 10.4 vs 8.5) and colder; regress total_err_mkt ~ div + season FE + week FE + weather;
(c) rolling-origin 2014-2025 of the div totals gap applied to the market and to a rating-based total proxy (the model's
engine: prior-mean + 0.35*rating_sum) with and without the recommended -1.0/+0.5; (d) season sign tests."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from kit import merged
pd.set_option("display.width", 220)
rng = np.random.default_rng(55)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m["rating_sum"] = m.home_pts_vs_avg + m.away_pts_vs_avg
m = m.dropna(subset=["rating_dif","mkt_spread","mkt_total"]).copy(); m["r"] = m.margin - m.rating_dif
m["div"] = m.div_game.astype(int); m["late"] = (m.week>=14).astype(int)
m["dome"] = m.is_dome.astype(int); m["temp_f"] = np.where(m.is_dome, 70.0, m.temp); m["wind_m"] = np.where(m.is_dome, 0.0, m.wind)
m["temp_f"] = m.temp_f.fillna(m.temp_f.mean()); m["wind_m"] = m.wind_m.fillna(m.wind_m.mean()); m["cold"] = (m.temp_f<40).astype(int); m["windy"] = (m.wind_m>=15).astype(int)
x = m[m.season!=2020].copy(); fit = m[(m.season<=2021)&(m.season!=2020)].copy(); test = m[m.season>=2022].copy()

# (a) spread rolling-origin
recs = []
for s in [s for s in range(2014, 2026) if s != 2020]:
    f = m[(m.season<s)&(m.season!=2020)]; te = m[m.season==s].copy(); ps = sorted(set(f.season)); f5 = f[f.season.isin(ps[-5:])]
    te["K"] = f5.r.mean(); te["Kdiv"] = np.where(te["div"]==1, f5[f5["div"]==1].r.mean(), f5[f5["div"]==0].r.mean()); recs.append(te)
T = pd.concat(recs); e0 = T.r - T.K; e1 = T.r - T.Kdiv
o = sm.OLS(e0, sm.add_constant(T.Kdiv - T.K)).fit(cov_type="HC1")
print("(a) SPREAD rolling-origin 2014-25 ex 2020 (n=%d): MAE const=%.3f, div-specific HFA=%.3f; residual ~ (div adj) slope=%.2f (se %.2f, p=%.3f); mean div adj in div games=%.2f" %
      (len(T), e0.abs().mean(), e1.abs().mean(), o.params.iloc[1], o.bse.iloc[1], o.pvalues.iloc[1], (T.Kdiv-T.K)[T["div"]==1].mean()))

# (b) totals confounds
print("\n(b) TOTALS: market total residual (total_pts - total_line) ~ div with controls, HC1")
specs = {"div + season FE": "total_err_mkt ~ div + C(season)",
         "+ week FE": "total_err_mkt ~ div + C(season) + C(week)",
         "+ week FE + dome + temp + wind": "total_err_mkt ~ div + C(season) + C(week) + dome + temp_f + wind_m",
         "+ week FE + cold + windy + dome": "total_err_mkt ~ div + C(season) + C(week) + dome + cold + windy"}
for lab, dd in [("FULL 2009-25 ex2020", x), ("FIT 2009-21 ex2020", fit), ("TEST 2022-25", test)]:
    out = []
    for nm, fml in specs.items():
        o = smf.ols(fml, data=dd).fit(cov_type="HC1"); out.append("%s: %+.2f (se %.2f, p=%.3f)" % (nm, o.params["div"], o.bse["div"], o.pvalues["div"]))
    print("  " + lab + "\n     " + "\n     ".join(out))
print("  raw total ~ div + season FE + week FE + weather (FULL): div=%+.2f (p=%.3f)" % ((lambda o: (o.params["div"], o.pvalues["div"]))(smf.ols("total_pts ~ div + C(season) + C(week) + dome + temp_f + wind_m", data=x).fit(cov_type="HC1"))))
print("  market total ~ div + season FE + week FE + weather (FULL): div=%+.2f (p=%.3f)  [how much the market already prices]" % ((lambda o: (o.params["div"], o.pvalues["div"]))(smf.ols("mkt_total ~ div + C(season) + C(week) + dome + temp_f + wind_m", data=x).fit(cov_type="HC1"))))
print("  div gap in market total residual by week bucket (FULL ex 2020):")
for lab, lo, hi in [("wk 1-8",1,8),("wk 9-13",9,13),("wk 14-18",14,18)]:
    dd = x[(x.week>=lo)&(x.week<=hi)]; a, b = dd[dd["div"]==1].total_err_mkt, dd[dd["div"]==0].total_err_mkt
    print("    %-8s n_div=%4d n_non=%4d  div resid %+.2f  non-div %+.2f  gap %+.2f (p=%.3f)" % (lab, len(a), len(b), a.mean(), b.mean(), a.mean()-b.mean(), stats.ttest_ind(a,b,equal_var=False).pvalue))
print("  and by roof (FULL ex 2020):")
for lab, dd in [("dome/closed", x[x.is_dome]), ("outdoor", x[~x.is_dome])]:
    a, b = dd[dd["div"]==1].total_err_mkt, dd[dd["div"]==0].total_err_mkt
    print("    %-11s gap %+.2f (p=%.3f) n=%d/%d" % (lab, a.mean()-b.mean(), stats.ttest_ind(a,b,equal_var=False).pvalue, len(a), len(b)))
# late-season non-div games: is the market residual also negative? (isolates 'late' from 'div')
print("  market total residual, non-div games by week bucket:", {lab: round(x[(x["div"]==0)&(x.week>=lo)&(x.week<=hi)].total_err_mkt.mean(),2) for lab,lo,hi in [("1-8",1,8),("9-13",9,13),("14-18",14,18)]})
print("  market total residual, div games by week bucket:    ", {lab: round(x[(x["div"]==1)&(x.week>=lo)&(x.week<=hi)].total_err_mkt.mean(),2) for lab,lo,hi in [("1-8",1,8),("9-13",9,13),("14-18",14,18)]})
# what the MODEL engine would see: raw total controlling for rating sum
o = smf.ols("total_pts ~ rating_sum + div + C(season)", data=x).fit(cov_type="HC1")
print("  raw total ~ 0.35-type rating_sum + div + season FE (FULL): rating_sum b=%.2f, div=%+.2f (se %.2f, p=%.3f)" % (o.params["rating_sum"], o.params["div"], o.bse["div"], o.pvalues["div"]))

# (c) rolling-origin totals
recs = []
for s in [s for s in range(2014, 2026) if s != 2020]:
    f = m[(m.season<s)&(m.season!=2020)]; te = m[m.season==s].copy(); ps = sorted(set(f.season)); f5 = f[f.season.isin(ps[-5:])]
    o = smf.ols("total_err_mkt ~ div + C(season)", data=f).fit(); te["coef_all"] = o.params["div"]
    o5 = smf.ols("total_err_mkt ~ div + C(season)", data=f5).fit(); te["coef5"] = o5.params["div"]
    # model-engine proxy: prior-season mean total + 0.35*rating_sum ; and fitted slope version
    last = f[f.season==ps[-1]]; te["tot_hat"] = last.total_pts.mean() + 0.35*te.rating_sum
    ob = smf.ols("total_pts ~ rating_sum", data=f5).fit(); te["tot_hat_fit"] = ob.params["Intercept"] + ob.params["rating_sum"]*te.rating_sum + (last.total_pts.mean() - f5.total_pts.mean())
    recs.append(te)
T = pd.concat(recs); n = len(T); idx = rng.integers(0, n, (3000, n))
print("\n(c) TOTALS rolling-origin 2014-25 ex 2020 (n=%d, div n=%d). coef_all by season:" % (n, T["div"].sum()), T.groupby("season").coef_all.first().round(2).to_dict())
gap = T[T["div"]==1].total_err_mkt.mean() - T[T["div"]==0].total_err_mkt.mean()
print("    pooled OOS market residual gap div - non-div = %+.2f (Welch p=%.3f); div under rate %.3f (n=%d)" %
      (gap, stats.ttest_ind(T[T["div"]==1].total_err_mkt, T[T["div"]==0].total_err_mkt, equal_var=False).pvalue, (T[T["div"]==1].total_err_mkt<0).sum()/(T[T["div"]==1].total_err_mkt!=0).sum(), T["div"].sum()))
ci = stats.binomtest(int((T[T["div"]==1].total_err_mkt<0).sum()), int((T[T["div"]==1].total_err_mkt!=0).sum())).proportion_ci(0.95); print("    div under-rate 95%% CI [%.3f, %.3f]" % ci)
adj_rec = np.where(T["div"]==1, -1.0, 0.5)
def rep(nm, pred):
    e = T.total_pts.values - pred; return nm, e
base_m = T.mkt_total.values; base_h = T.tot_hat.values; base_hf = T.tot_hat_fit.values
rows = []
for nm, pred, base in [("market", base_m, base_m), ("market + coef_all*div", base_m + T.coef_all.values*T["div"].values, base_m), ("market + coef5*div", base_m + T.coef5.values*T["div"].values, base_m),
                       ("market + (-1.0 div / +0.5 non)", base_m + adj_rec, base_m), ("market + (-1.0 div / 0 non)", base_m + np.where(T["div"]==1,-1.0,0.0), base_m),
                       ("engine proxy (prior mean + 0.35*rsum)", base_h, base_h), ("engine proxy + (-1.0/+0.5)", base_h + adj_rec, base_h), ("engine proxy + (-1.2/+0.65 raw gap)", base_h + np.where(T["div"]==1,-1.2,0.65), base_h),
                       ("engine proxy fitted slope", base_hf, base_hf), ("engine proxy fitted + (-1.0/+0.5)", base_hf + adj_rec, base_hf)]:
    e = T.total_pts.values - pred; e_b = T.total_pts.values - base; dd = np.abs(e) - np.abs(e_b); bs = dd[idx].mean(axis=1)
    rows.append(dict(spec=nm, MAE=np.abs(e).mean(), bias_div=e[T["div"]==1].mean(), bias_non=e[T["div"]==0].mean(), dMAE_vs_base=dd.mean(), ci_lo=np.percentile(bs,2.5), ci_hi=np.percentile(bs,97.5)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# O/U vs market of the engine proxy with/without the adjustment: does the adjustment change pick direction usefully?
for nm, pred in [("engine proxy", base_h), ("engine proxy + (-1.0/+0.5)", base_h + adj_rec)]:
    pick_over = pred > base_m; res = np.sign(T.total_pts.values - base_m); w = ((pick_over & (res>0)) | (~pick_over & (res<0))).sum(); l = (res!=0).sum() - w
    print("    %-28s O/U vs market: %d-%d (%.3f)" % (nm, w, l, w/(w+l)))

# (d) season sign tests
S = x.groupby("season").apply(lambda d: d[d["div"]==1].total_err_mkt.mean() - d[d["div"]==0].total_err_mkt.mean())
print("\n(d) seasons (ex 2020) with negative div residual gap: %d of %d, binomial p=%.3f; mean gap %+.2f (se %.2f, t=%.2f); 2022-25: %s" %
      ((S<0).sum(), len(S), stats.binomtest(int((S<0).sum()), len(S)).pvalue, S.mean(), S.std()/np.sqrt(len(S)), S.mean()/(S.std()/np.sqrt(len(S))), S.loc[2022:].round(2).to_dict()))
Sw = x.groupby("season").apply(lambda d: smf.ols("total_err_mkt ~ div + C(week)", data=d).fit().params["div"])
print("    same with within-season week FE: negative in %d of %d, mean %+.2f (se %.2f)" % ((Sw<0).sum(), len(Sw), Sw.mean(), Sw.std()/np.sqrt(len(Sw))))
