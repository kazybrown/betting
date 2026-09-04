"""THEORY 3: HFA by slot (Sunday early/late, SNF, MNF, TNF, Saturday, other) and Week 1, vs Sunday afternoon.
Rating-adjusted residual = margin - rating_dif (nfelo QB-adj Elo diff / 25). Market residual = margin + mkt_spread.
Fit slot coefficients on 2009-2021 (2020 excl), test 2022-2025."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from kit import merged
pd.set_option("display.width", 220)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy()
m["hfa_resid"] = m.margin - m.rating_dif
m["hour"] = m.gametime.str.slice(0,2).astype(int) + m.gametime.str.slice(3,5).astype(int)/60
def slot(r):
    if r.weekday == "Sunday":
        return "SUN_early" if r.hour < 15 else ("SUN_late" if r.hour < 19 else "SNF")
    if r.weekday == "Monday": return "MNF"
    if r.weekday == "Thursday": return "TNF"
    if r.weekday == "Saturday": return "SAT"
    return "OTHER"
m["slot"] = m.apply(slot, axis=1)
m["sun_aft"] = m.slot.isin(["SUN_early","SUN_late"])
m["wk1"] = m.week.eq(1)
m["thanks"] = m.slot.eq("TNF") & m.week.isin([12,13]) & (m.hour < 18)
m["rest_dif"] = m.home_rest - m.away_rest
m["mkt_hfa"] = m.spread_line - m.rating_dif

def table(d, label):
    rows = []
    for s, x in d.groupby("slot"):
        rows.append(dict(slot=s, n=len(x), hfa=x.hfa_resid.mean(), hfa_se=x.hfa_resid.std()/np.sqrt(len(x)),
                         mkt_impl=x.mkt_hfa.mean(), mkt_resid=x.spread_err_mkt.mean(), mkt_se=x.spread_err_mkt.std()/np.sqrt(len(x)),
                         home_cover=(x.spread_err_mkt>0).sum()/(x.spread_err_mkt!=0).sum(), nfelo_hfa=x.hfa_pts.mean(), rest_dif=x.rest_dif.mean()))
    for nm, mask in [("WEEK1", d.wk1), ("Thanksgiving", d.thanks), ("SUN_aft (ref)", d.sun_aft)]:
        x = d[mask]
        rows.append(dict(slot=nm, n=len(x), hfa=x.hfa_resid.mean(), hfa_se=x.hfa_resid.std()/np.sqrt(len(x)),
                         mkt_impl=x.mkt_hfa.mean(), mkt_resid=x.spread_err_mkt.mean(), mkt_se=x.spread_err_mkt.std()/np.sqrt(len(x)),
                         home_cover=(x.spread_err_mkt>0).sum()/(x.spread_err_mkt!=0).sum(), nfelo_hfa=x.hfa_pts.mean(), rest_dif=x.rest_dif.mean()))
    T = pd.DataFrame(rows).set_index("slot")
    ref = d[d.sun_aft]
    T["p_vs_sunaft"] = [stats.ttest_ind(d[(d.slot==s) if s in d.slot.unique() else (d.wk1 if s=="WEEK1" else d.thanks if s=="Thanksgiving" else d.sun_aft)].hfa_resid, ref.hfa_resid, equal_var=False).pvalue for s in T.index]
    print(f"\n{label}: rating-adjusted HFA by slot (hfa), market-implied (mkt_impl), market residual (+ = home under-priced)")
    print(T.round(3).to_string())
    return T
full = m[m.season!=2020]
T_full = table(full, "2009-2025 ex 2020")
T_rec = table(m[m.season>=2021], "2021-2025")

# --- regression with season FE, fit window, then OOS ---
fit = m[(m.season<=2021)&(m.season!=2020)].copy(); test = m[m.season>=2022].copy()
fit["slotc"] = pd.Categorical(fit.slot, categories=["SUN_early","SUN_late","SNF","MNF","TNF","SAT","OTHER"])
mod = smf.ols("hfa_resid ~ C(slotc) + wk1 + C(season)", data=fit).fit(cov_type="HC1")
print("\nFit 2009-2021: hfa_resid ~ slot + week1 + season FE (ref = SUN_early)")
co = mod.params.filter(like="slotc").rename(lambda s: s.replace("C(slotc)[T.","").replace("]","")); se = mod.bse.filter(like="slotc"); pv = mod.pvalues.filter(like="slotc")
for k, v, s_, p in zip(co.index, co.values, se.values, pv.values): print("  %-10s %+.2f (se %.2f, p=%.3f)" % (k, v, s_, p))
print("  wk1        %+.2f (se %.2f, p=%.3f)" % (mod.params["wk1[T.True]"], mod.bse["wk1[T.True]"], mod.pvalues["wk1[T.True]"]))
# also with rest_dif control
mod2 = smf.ols("hfa_resid ~ C(slotc) + wk1 + rest_dif + C(season)", data=fit).fit(cov_type="HC1")
print("  with rest_dif control: TNF %+.2f (p=%.3f), MNF %+.2f, SNF %+.2f, rest_dif %+.2f/day (p=%.3f)" %
      (mod2.params["C(slotc)[T.TNF]"], mod2.pvalues["C(slotc)[T.TNF]"], mod2.params["C(slotc)[T.MNF]"], mod2.params["C(slotc)[T.SNF]"], mod2.params["rest_dif"], mod2.pvalues["rest_dif"]))
# market residual model in fit window
modm = smf.ols("spread_err_mkt ~ C(slotc) + wk1 + C(season)", data=fit).fit(cov_type="HC1")
print("Fit 2009-2021: market residual ~ slot: " + ", ".join("%s %+.2f (p=%.2f)" % (k.replace("C(slotc)[T.","").replace("]",""), v, p) for k, v, p in zip(modm.params.filter(like="slotc").index, modm.params.filter(like="slotc"), modm.pvalues.filter(like="slotc"))) + ", wk1 %+.2f (p=%.2f)" % (modm.params["wk1[T.True]"], modm.pvalues["wk1[T.True]"]))

# OOS: slot adjustment from fit window applied to constant-HFA spread
K = fit.hfa_resid.mean()
adj = {"SUN_early":0.0, **{k: v for k, v in co.items()}}
# recentre so that the SUN_early-referenced coefficients become deviations from the fit-window mean
fit_slot_mean = fit.slot.map(adj).mean()
test["slot_adj"] = test.slot.map(adj) - fit_slot_mean + (mod.params["wk1[T.True]"]*test.wk1 - (mod.params["wk1[T.True]"]*fit.wk1).mean())
e0 = test.margin - test.rating_dif - K; e1 = e0 - test.slot_adj
print("\nOOS 2022-2025 (n=%d): MAE const=%.3f | const+slot adj=%.3f | bias %+.3f vs %+.3f" % (len(test), e0.abs().mean(), e1.abs().mean(), e0.mean(), e1.mean()))
ols = sm.OLS(e0, sm.add_constant(test.slot_adj)).fit(cov_type="HC1")
print("OOS residual ~ fitted slot adj: slope=%.2f (se %.2f, p=%.3f) [1=predictive]" % (ols.params.iloc[1], ols.bse.iloc[1], ols.pvalues.iloc[1]))
olsm = sm.OLS(test.spread_err_mkt, sm.add_constant(test.slot_adj)).fit(cov_type="HC1")
print("OOS MARKET residual ~ fitted slot adj: slope=%.2f (se %.2c, p=%.3f)" % (olsm.params.iloc[1], olsm.bse.iloc[1], olsm.pvalues.iloc[1]) if False else
      "OOS MARKET residual ~ fitted slot adj: slope=%.2f (se %.2f, p=%.3f)" % (olsm.params.iloc[1], olsm.bse.iloc[1], olsm.pvalues.iloc[1]))
# Simple OOS check per slot: sign agreement of fit-window and test-window deviations
print("\nSlot deviation from Sunday-afternoon HFA: fit window vs test window (points)")
ref_f = fit[fit.sun_aft].hfa_resid.mean(); ref_t = test[test.sun_aft].hfa_resid.mean()
for s in ["SNF","MNF","TNF","SAT"]:
    a = fit[fit.slot==s].hfa_resid; b = test[test.slot==s].hfa_resid
    print("  %-4s fit: %+.2f (n=%d)  test: %+.2f (n=%d, se %.2f)" % (s, a.mean()-ref_f, len(a), b.mean()-ref_t, len(b), b.std()/np.sqrt(len(b))))
a = fit[fit.wk1].hfa_resid; b = test[test.wk1].hfa_resid
print("  WK1  fit: %+.2f (n=%d)  test: %+.2f (n=%d, se %.2f)" % (a.mean()-ref_f, len(a), b.mean()-ref_t, len(b), b.std()/np.sqrt(len(b))))
# Market ATS by slot OOS (home cover rate 2022-25) with binomial CI
print("\nOOS 2022-25 home cover rate vs market by slot:")
for s, x in test.groupby("slot"):
    w = (x.spread_err_mkt>0).sum(); nn = (x.spread_err_mkt!=0).sum(); lo, hi = stats.binomtest(int(w), int(nn)).proportion_ci(0.95)
    print("  %-10s %d/%d = %.3f  CI[%.3f, %.3f]" % (s, w, nn, w/nn, lo, hi))
