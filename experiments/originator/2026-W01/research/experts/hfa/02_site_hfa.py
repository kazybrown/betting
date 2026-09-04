"""THEORY 2: Site-specific HFA with empirical-Bayes shrinkage; does nfelo hfa capture it?
Unit = home franchise (kit-normalized ids). Residual r = margin - rating_dif - leagueHFA(season)
(league HFA = season mean of margin - rating_dif over all home REG games; in the fit window only).
Fit window 2009-2021 (2020 excluded), test 2022-2025."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from kit import merged
pd.set_option("display.width", 220)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy()
m["hfa_resid"] = m.margin - m.rating_dif
m["nfelo_hfa"] = m.hfa_pts; m["nfelo_hfa_base"] = m.hfa_mod/25
m["tz_adv"] = m.home_time_advantage_mod.fillna(0)/25

fit = m[(m.season<=2021)&(m.season!=2020)].copy(); test = m[m.season>=2022].copy()
lg_fit = fit.groupby("season").hfa_resid.transform("mean"); fit["r"] = fit.hfa_resid - lg_fit
K_LEAGUE = fit.hfa_resid.mean()   # constant used OOS
test["r"] = test.hfa_resid - K_LEAGUE
print("fit n=%d test n=%d  K_LEAGUE(fit)=%.2f" % (len(fit), len(test), K_LEAGUE))

def eb(df, col="r"):
    g = df.groupby("home")[col].agg(["mean","count","std"]); g["se2"] = g["std"]**2/g["count"]
    tau2 = max(0.0, g["mean"].var(ddof=1) - g["se2"].mean())
    g["shrink"] = tau2/(tau2+g["se2"]); g["eb"] = g["shrink"]*g["mean"]
    return g, tau2
G, tau2 = eb(fit)
print("Between-team variance of true HFA excess (method of moments) tau^2=%.3f -> tau=%.2f pts; mean shrink factor=%.2f" % (tau2, np.sqrt(tau2), G.shrink.mean()))
Gt, tau2_t = eb(test)
G["test_mean"] = Gt["mean"]; G["test_n"] = Gt["count"]
G["nfelo_hfa_fit"] = fit.groupby("home").nfelo_hfa.mean() - fit.nfelo_hfa.mean()
G["nfelo_hfa_test"] = test.groupby("home").nfelo_hfa.mean() - test.nfelo_hfa.mean()
G["tz_adv_test"] = test.groupby("home").tz_adv.mean()
print("\nPer-team HFA excess over league (points). raw=fit-window mean, eb=shrunk, test_mean=2022-25 realized, nfelo_* = nfelo's hfa_pts minus league mean")
print(G[["count","mean","se2","shrink","eb","test_mean","test_n","nfelo_hfa_fit","nfelo_hfa_test","tz_adv_test"]].rename(columns={"mean":"raw","count":"n_fit"}).sort_values("eb", ascending=False).round(2).to_string())
print("\nTest-window tau^2=%.3f (tau=%.2f)" % (tau2_t, np.sqrt(max(tau2_t,0))))

# ---- OOS predictiveness ----
test = test.join(G[["eb","mean"]].rename(columns={"mean":"raw_fit"}), on="home")
for nm, col in [("EB-shrunk fit excess","eb"), ("raw fit excess","raw_fit"), ("nfelo hfa_pts - leaguemean (test)", None)]:
    x = test[col] if col else (test.nfelo_hfa - test.nfelo_hfa.mean())
    ols = sm.OLS(test.r, sm.add_constant(x)).fit(cov_type="HC1")
    print("OOS 2022-25: realized excess ~ %-34s slope=%.2f (se %.2f, p=%.3f)  [1 = fully predictive, 0 = noise]" % (nm, ols.params.iloc[1], ols.bse.iloc[1], ols.pvalues.iloc[1]))
# team-level split-half correlation
ok = G.dropna(subset=["test_mean"])
print("Team-level corr(fit excess, test excess): r=%.2f (n=%d teams, p=%.3f) ; corr(EB, test): r=%.2f" %
      (stats.pearsonr(ok["mean"], ok["test_mean"])[0], len(ok), stats.pearsonr(ok["mean"], ok["test_mean"])[1], stats.pearsonr(ok["eb"], ok["test_mean"])[0]))
print("Team-level corr(nfelo hfa excess test, realized test excess): r=%.2f (p=%.3f)" % stats.pearsonr(ok["nfelo_hfa_test"], ok["test_mean"]))
# OOS MAE: constant league HFA vs + EB team excess vs nfelo per-game
def rep(nm, pred):
    e = test.margin + pred; return dict(spec=nm, MAE=e.abs().mean(), RMSE=np.sqrt((e**2).mean()), bias=e.mean())
R = pd.DataFrame([rep("const K_LEAGUE", -(test.rating_dif + K_LEAGUE)),
                  rep("const + EB team excess", -(test.rating_dif + K_LEAGUE + test.eb)),
                  rep("const + raw team excess", -(test.rating_dif + K_LEAGUE + test.raw_fit)),
                  rep("const + 0.5*EB", -(test.rating_dif + K_LEAGUE + 0.5*test.eb)),
                  rep("nfelo hfa_pts per game", -(test.rating_dif + test.nfelo_hfa)),
                  rep("nfelo hfa_mod/25 per game", -(test.rating_dif + test.nfelo_hfa_base)),
                  rep("const + nfelo tz_adv only", -(test.rating_dif + K_LEAGUE + test.tz_adv))])
print("\nOOS 2022-2025 spread error (n=%d):" % len(test)); print(R.round(3).to_string(index=False))
# paired bootstrap for EB vs const
rng = np.random.default_rng(3); n=len(test); e0 = (test.margin - test.rating_dif - K_LEAGUE).abs().values; e1 = (test.margin - test.rating_dif - K_LEAGUE - test.eb).abs().values
d = np.array([ (e1[i]-e0[i]).mean() for i in (rng.integers(0,n,n) for _ in range(3000))])
print("paired bootstrap MAE(EB) - MAE(const): %+.3f  95%% CI [%+.3f, %+.3f]" % ((e1-e0).mean(), np.percentile(d,2.5), np.percentile(d,97.5)))

# ---- vs MARKET: does fit-window team excess predict market residual OOS? ----
ols = sm.OLS(test.spread_err_mkt, sm.add_constant(test.eb)).fit(cov_type="HC1")
print("\nOOS market residual ~ EB team excess: slope=%.2f (se %.2f, p=%.3f)" % (ols.params.iloc[1], ols.bse.iloc[1], ols.pvalues.iloc[1]))
# market-implied team HFA (intercept by team of spread_line - rating_dif) vs realized, test window
test["mkt_hfa"] = test.spread_line - test.rating_dif
mk = test.groupby("home").agg(mkt_hfa=("mkt_hfa","mean"), realized=("hfa_resid","mean"), nfelo=("nfelo_hfa","mean"), n=("r","size"))
mk["mkt_excess"] = mk.mkt_hfa - mk.mkt_hfa.mean(); mk["real_excess"] = mk.realized - mk.realized.mean(); mk["nfelo_excess"] = mk.nfelo - mk.nfelo.mean()
print("corr(market-implied team HFA excess, nfelo excess) 2022-25: r=%.2f ; corr(market excess, realized excess): r=%.2f" %
      (stats.pearsonr(mk.mkt_excess, mk.nfelo_excess)[0], stats.pearsonr(mk.mkt_excess, mk.real_excess)[0]))
print(mk.sort_values("mkt_excess", ascending=False).round(2).to_string())

# ---- Named venues, full-sample descriptive (2009-2025 excl 2020) ----
full = m[m.season!=2020].copy(); lg = full.groupby("season").hfa_resid.transform("mean"); full["r"] = full.hfa_resid - lg
Gf, tau2f = eb(full)
print("\nFull-sample (2009-25 ex 2020) per-team excess, EB-shrunk; tau=%.2f" % np.sqrt(tau2f))
names = ["SEA","DEN","KC","GB","NO","BAL","PIT","NE","ARI","LAC","LA","LV","JAX","DAL","NYG","NYJ","MIA","TB","MIN","PHI","BUF","SF"]
print(Gf.loc[[t for t in names if t in Gf.index], ["count","mean","std","shrink","eb"]].round(2).to_string())
# nfelo calibration: regress realized (margin - rating_dif) on nfelo hfa_pts across all games, by era
for lo, hi in [(2009,2019),(2021,2025),(2022,2025)]:
    d = m[(m.season>=lo)&(m.season<=hi)]
    o = sm.OLS(d.hfa_resid, sm.add_constant(d.nfelo_hfa)).fit(cov_type="HC1")
    o2 = sm.OLS(d.hfa_resid, sm.add_constant(d.nfelo_hfa_base)).fit(cov_type="HC1")
    print("nfelo calibration %d-%d: resid ~ hfa_pts slope=%.2f (se %.2f) icpt=%.2f | ~ hfa_mod/25 slope=%.2f (se %.2f)" %
          (lo,hi,o.params.iloc[1],o.bse.iloc[1],o.params.iloc[0],o2.params.iloc[1],o2.bse.iloc[1]))
