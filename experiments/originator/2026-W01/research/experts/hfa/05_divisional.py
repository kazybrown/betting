"""THEORY 5: Divisional games -- smaller HFA? smaller margins? does the market already price it?
Fit 2009-2021 (2020 excl), test 2022-2025."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from kit import merged, ats
pd.set_option("display.width", 220)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy()
m["hfa_resid"] = m.margin - m.rating_dif; m["mkt_hfa"] = m.spread_line - m.rating_dif
m["div"] = m.div_game.astype(int); m["absm"] = m.margin.abs(); m["mkt_abs_err"] = m.spread_err_mkt.abs()
m["fav_margin_vs_line"] = np.where(m.spread_line>0, m.margin - m.spread_line, -(m.margin - m.spread_line))  # + = favourite covered
m["fav_cover"] = np.sign(m.fav_margin_vs_line)
m["abs_line"] = m.spread_line.abs()

def desc(d, label):
    rows = []
    for dv, x in d.groupby("div"):
        rows.append(dict(div=dv, n=len(x), hfa=x.hfa_resid.mean(), hfa_se=x.hfa_resid.std()/np.sqrt(len(x)), mkt_impl=x.mkt_hfa.mean(),
                         mkt_resid=x.spread_err_mkt.mean(), mkt_se=x.spread_err_mkt.std()/np.sqrt(len(x)),
                         home_cover=(x.spread_err_mkt>0).sum()/(x.spread_err_mkt!=0).sum(),
                         mean_abs_margin=x.absm.mean(), sd_margin=x.margin.std(), mkt_MAE=x.mkt_abs_err.mean(),
                         abs_line=x.abs_line.mean(), abs_rating_dif=x.rating_dif.abs().mean(),
                         fav_cover=(x.fav_cover>0).sum()/(x.fav_cover!=0).sum(), nfelo_hfa=x.hfa_pts.mean(), nfelo_div_mod=(x.div_game_mod/25).mean()))
    T = pd.DataFrame(rows).set_index("div"); print(f"\n{label}"); print(T.round(3).to_string())
    a, b = d[d["div"]==1], d[d["div"]==0]
    print("  Welch p: hfa %.3f | mkt_resid %.3f | abs margin %.3f | mkt MAE %.3f" % (
        stats.ttest_ind(a.hfa_resid, b.hfa_resid, equal_var=False).pvalue, stats.ttest_ind(a.spread_err_mkt, b.spread_err_mkt, equal_var=False).pvalue,
        stats.ttest_ind(a.absm, b.absm, equal_var=False).pvalue, stats.ttest_ind(a.mkt_abs_err, b.mkt_abs_err, equal_var=False).pvalue))
    return T
desc(m[m.season!=2020], "ALL 2009-2025 ex 2020")
desc(m[(m.season<=2021)&(m.season!=2020)], "FIT 2009-2021 ex 2020")
desc(m[m.season>=2022], "TEST 2022-2025")

fit = m[(m.season<=2021)&(m.season!=2020)].copy(); test = m[m.season>=2022].copy()
# what does nfelo's div_game_mod do?
print("\nnfelo div_game_mod (Elo) in div games: by sign of elo dif:", fit[fit["div"]==1].groupby(np.sign(fit[fit["div"]==1].elo_dif_pts)).div_game_mod.mean().round(1).to_dict(),
      "| corr with elo_dif:", round(np.corrcoef(fit[fit["div"]==1].div_game_mod, fit[fit["div"]==1].elo_dif_pts)[0,1],2))
# compression: margin ~ rating_dif * div (+ season FE)
mod = smf.ols("margin ~ rating_dif*div + C(season)", data=fit).fit(cov_type="HC1")
print("\nFIT: margin ~ rating_dif*div + season FE: div HFA shift=%+.2f (se %.2f, p=%.3f); rating slope=%.2f; slope x div=%+.2f (se %.2f, p=%.3f)" %
      (mod.params["div"], mod.bse["div"], mod.pvalues["div"], mod.params["rating_dif"], mod.params["rating_dif:div"], mod.bse["rating_dif:div"], mod.pvalues["rating_dif:div"]))
modm = smf.ols("margin ~ spread_line*div + C(season)", data=fit).fit(cov_type="HC1")
print("FIT: margin ~ spread_line*div: div shift=%+.2f (p=%.3f); market slope=%.2f; slope x div=%+.2f (se %.2f, p=%.3f)" %
      (modm.params["div"], modm.pvalues["div"], modm.params["spread_line"], modm.params["spread_line:div"], modm.bse["spread_line:div"], modm.pvalues["spread_line:div"]))
modt = smf.ols("margin ~ spread_line*div + C(season)", data=test).fit(cov_type="HC1")
print("TEST(descriptive): margin ~ spread_line*div: div shift=%+.2f (p=%.3f); slope=%.2f; slope x div=%+.2f (se %.2f, p=%.3f)" %
      (modt.params["div"], modt.pvalues["div"], modt.params["spread_line"], modt.params["spread_line:div"], modt.bse["spread_line:div"], modt.pvalues["spread_line:div"]))
# OOS: apply fit-window div adjustments to a rating spread
K = fit.hfa_resid.mean(); dshift = mod.params["div"]; dslope = mod.params["rating_dif:div"]; base_slope = mod.params["rating_dif"]
hfa_div = fit[fit["div"]==1].hfa_resid.mean(); hfa_non = fit[fit["div"]==0].hfa_resid.mean()
specs = {"const K": -(test.rating_dif + K),
         "div-specific HFA (fit)": -(test.rating_dif + np.where(test["div"]==1, hfa_div, hfa_non)),
         "div HFA + slope compression (fit)": -(test.rating_dif*(1 + dslope/base_slope*test["div"]) + np.where(test["div"]==1, hfa_div, hfa_non)),
         "nfelo close line": test.nfelo_home_line_close}
print("\nOOS 2022-25 (n=%d, div n=%d):" % (len(test), test["div"].sum()))
for nm, pred in specs.items():
    e = test.margin + pred
    for dv in [0,1]:
        ee = e[test["div"]==dv]; print("  %-36s div=%d MAE=%.3f bias=%+.3f" % (nm, dv, ee.abs().mean(), ee.mean()))
# favourite cover rates div vs non-div, fit & test, with binomial CIs
print("\nFavourite ATS cover rate vs closing line (pushes excluded):")
for lab, d in [("fit 2009-21", fit), ("test 2022-25", test)]:
    for dv in [0,1]:
        x = d[(d["div"]==dv) & (d.fav_cover!=0)]; w = (x.fav_cover>0).sum(); n = len(x); ci = stats.binomtest(int(w), int(n)).proportion_ci(0.95)
        print("  %-12s div=%d  %d/%d = %.3f CI[%.3f, %.3f]" % (lab, dv, w, n, w/n, ci[0], ci[1]))
# by favourite size within div games (test)
print("\nTEST div games by |line| bucket: favourite cover rate and mean fav margin vs line")
for lab, d in [("fit", fit), ("test", test)]:
    dd = d[d["div"]==1]; dd = dd.assign(b=pd.cut(dd.abs_line, [-0.1, 3, 6.5, 30], labels=["0-3","3.5-6.5","7+"]))
    for b_, x in dd.groupby("b", observed=True):
        xx = x[x.fav_cover!=0]; print("  %-4s |line| %-7s n=%3d fav cover=%.3f  fav margin vs line=%+.2f (se %.2f)" % (lab, b_, len(x), (xx.fav_cover>0).mean(), x.fav_margin_vs_line.mean(), x.fav_margin_vs_line.std()/np.sqrt(len(x))))
# total points in div games (for totals model): mean total, market total residual
print("\nTotals: div vs non-div (all ex 2020): ")
for dv, x in m[m.season!=2020].groupby("div"):
    print("  div=%d mean total=%.2f  mkt total=%.2f  resid=%+.2f (se %.2f)" % (dv, x.total_pts.mean(), x.mkt_total.mean(), x.total_err_mkt.mean(), x.total_err_mkt.std()/np.sqrt(len(x))))
