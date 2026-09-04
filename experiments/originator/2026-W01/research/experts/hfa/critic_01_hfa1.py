"""CRITIC HFA-1: re-check league HFA level/trend and the OOS constant choice.
Attacks: (a) rating-free estimates (mean/median margin, mean spread_line) so the level does not hinge on nfelo's scale;
(b) median-regression intercept -- an MAE-optimal constant is a median, not a mean;
(c) era tests with block boundaries not chosen post hoc (5-season blocks) plus trend tests on raw/median margin;
(d) PRE-SPECIFIED rolling-origin HFA constants scored 2015-2025 and 2022-2025 vs fixed 1.75/2.0/2.5/nfelo per game,
    with paired bootstrap CIs and ATS vs market; (e) show that the expert's k grid was selected on the test window."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from kit import merged, ats
pd.set_option("display.width", 220)
rng = np.random.default_rng(2026)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m["rating_dif_noqb"] = m.elo_dif_pts
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy()
m["r"] = m.margin - m.rating_dif; m["r_noqb"] = m.margin - m.rating_dif_noqb
print("n=%d; sanity corr(mkt_spread, margin)=%.3f (must be strongly negative); corr(rating_dif, margin)=%.3f" %
      (len(m), np.corrcoef(m.mkt_spread, m.margin)[0,1], np.corrcoef(m.rating_dif, m.margin)[0,1]))

# ---------- (a)+(b) rating-free and median measures by block ----------
def blk(lo, hi):
    d = m[(m.season>=lo)&(m.season<=hi)&(m.season!=2020)]
    q = sm.QuantReg(d.margin, sm.add_constant(d.rating_dif)).fit(q=0.5)
    return dict(block=f"{lo}-{hi}", n=len(d), mean_margin=d.margin.mean(), median_margin=d.margin.median(),
                mean_line=d.spread_line.mean(), mean_rating_dif=d.rating_dif.mean(),
                hfa_mean=d.r.mean(), hfa_se=d.r.std()/np.sqrt(len(d)), hfa_median=d.r.median(),
                qreg_icpt=q.params["const"], qreg_se=q.bse["const"],
                hfa_noqb=d.r_noqb.mean(), mkt_resid_mean=d.spread_err_mkt.mean(), mkt_resid_median=d.spread_err_mkt.median(),
                home_cover=(d.spread_err_mkt>0).sum()/(d.spread_err_mkt!=0).sum(), home_win=(d.margin>0).mean())
B = pd.DataFrame([blk(2009,2013), blk(2014,2018), blk(2019,2021), blk(2021,2025), blk(2022,2025), blk(2009,2019), blk(2009,2021)])
print("\nBlocks (2020 excluded). hfa = margin - nfelo QB-adj Elo dif; qreg_icpt = median-regression intercept of margin ~ rating_dif")
print(B.round(2).to_string(index=False))

# ---------- (c) era tests ----------
def welch(a, b, la, lb):
    t = stats.ttest_ind(a, b, equal_var=False)
    print("  %s (%.2f, n=%d) vs %s (%.2f, n=%d): diff=%+.2f  p=%.3f" % (la, a.mean(), len(a), lb, b.mean(), len(b), a.mean()-b.mean(), t.pvalue))
x = m[m.season!=2020]
print("\nEra tests, rating-adjusted HFA (mean):")
welch(x[x.season<=2013].r, x[(x.season>=2014)&(x.season<=2019)].r, "2009-13", "2014-19")
welch(x[x.season<=2013].r, x[x.season>=2014].r, "2009-13", "2014-25")
welch(x[x.season<=2019].r, x[x.season>=2021].r, "2009-19", "2021-25")
print("Era tests, raw margin (rating-free):")
welch(x[x.season<=2013].margin, x[x.season>=2014].margin, "2009-13", "2014-25")
welch(x[x.season<=2019].margin, x[x.season>=2021].margin, "2009-19", "2021-25")
print("Era tests, market line (what the market priced):")
welch(x[x.season<=2019].spread_line, x[x.season>=2021].spread_line, "2009-19", "2021-25")
mt = stats.median_test(x[x.season<=2013].r, x[x.season>=2014].r)
print("Mood median test 2009-13 vs 2014-25 on rating-HFA: p=%.3f (medians %.2f vs %.2f)" % (mt[1], x[x.season<=2013].r.median(), x[x.season>=2014].r.median()))
T = x.groupby("season").agg(n=("margin","size"), raw=("margin","mean"), med=("margin","median"), r=("r","mean"), rmed=("r","median"), line=("spread_line","mean"))
print("Season-level WLS trends (2020 excl):")
for col in ["raw","med","r","rmed","line"]:
    w = sm.WLS(T[col], sm.add_constant(T.index.astype(float)), weights=T.n).fit()
    print("  %-5s slope=%+.3f/yr  p=%.3f" % (col, w.params.iloc[1], w.pvalues.iloc[1]))
# game-level trend with HC1 (season as continuous) - avoids the 16-point WLS
o = sm.OLS(x.r, sm.add_constant(x.season.astype(float))).fit(cov_type="HC1")
print("  game-level r ~ season: slope=%+.3f/yr (se %.3f, p=%.3f)" % (o.params.iloc[1], o.bse.iloc[1], o.pvalues.iloc[1]))

# ---------- (e) what could have been chosen pre-2022? ----------
fit = m[(m.season<=2021)&(m.season!=2020)]; test = m[m.season>=2022]
grid = np.arange(0, 3.51, 0.25)
mae_fit = [np.abs(fit.r - k).mean() for k in grid]; mae_test = [np.abs(test.r - k).mean() for k in grid]
print("\nFit-window (2009-21 ex 2020) MAE-optimal k on grid = %.2f (median r = %.2f, mean r = %.2f)" % (grid[int(np.argmin(mae_fit))], fit.r.median(), fit.r.mean()))
f5 = fit[fit.season>=2016]; print("Last-5-seasons (2016-21 ex 2020) MAE-optimal k = %.2f (median %.2f, mean %.2f)" % (grid[int(np.argmin([np.abs(f5.r-k).mean() for k in grid]))], f5.r.median(), f5.r.mean()))
print("TEST-window MAE-optimal k on grid = %.2f  <- the expert's 1.5-2.0 band and 1.75 pick were read off this test-window curve" % grid[int(np.argmin(mae_test))])
print("  test MAE by k:", {float(k): round(v,3) for k, v in zip(grid, mae_test)})

# ---------- (d) pre-specified rolling-origin ----------
def score(pred, d, label, store):
    e = d.margin.values + pred
    w, l, p = ats(pred, d.mkt_spread.values, d.margin.values)
    store[label] = e
    return dict(spec=label, n=len(e), MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), bias=e.mean(), ATS_vs_mkt=w/(w+l), w=w, l=l)
for lo in [2015, 2022]:
    seasons = [s for s in range(lo, 2026) if s != 2020]
    keys = ["roll5_mean","roll5_median","roll3_mean","allprior_mean","allprior_median","allprior_mean_incl2020","fixed1.5","fixed1.75","fixed2.0","fixed2.5","fixed3.0","nfelo_pergame","nfelo_hfa_mod","market"]
    parts = {k: [] for k in keys}; frames = []; kused = []
    for s in seasons:
        d = m[m.season==s]; frames.append(d)
        prior = m[(m.season<s)&(m.season!=2020)]; prior_all = m[m.season<s]
        ps = sorted(set(prior.season)); p5 = prior[prior.season.isin(ps[-5:])]; p3 = prior[prior.season.isin(ps[-3:])]
        kused.append(dict(season=s, roll5_mean=p5.r.mean(), roll5_median=p5.r.median(), roll3_mean=p3.r.mean(), allprior_mean=prior.r.mean(), allprior_median=prior.r.median(), incl2020=prior_all.r.mean()))
        parts["roll5_mean"].append(-(d.rating_dif + p5.r.mean())); parts["roll5_median"].append(-(d.rating_dif + p5.r.median()))
        parts["roll3_mean"].append(-(d.rating_dif + p3.r.mean()))
        parts["allprior_mean"].append(-(d.rating_dif + prior.r.mean())); parts["allprior_median"].append(-(d.rating_dif + prior.r.median()))
        parts["allprior_mean_incl2020"].append(-(d.rating_dif + prior_all.r.mean()))
        for k in [1.5,1.75,2.0,2.5,3.0]: parts[f"fixed{k}"].append(-(d.rating_dif + k))
        parts["nfelo_pergame"].append(-(d.rating_dif + d.hfa_pts)); parts["nfelo_hfa_mod"].append(-(d.rating_dif + d.hfa_mod/25)); parts["market"].append(d.mkt_spread)
    D = pd.concat(frames); store = {}
    R = pd.DataFrame([score(pd.concat(v).values, D, k, store) for k, v in parts.items()])
    print(f"\n=== Rolling-origin OOS {lo}-2025 ex 2020 (n={len(D)}) ===")
    print("k used per season:"); print(pd.DataFrame(kused).round(2).to_string(index=False))
    print(R.round(3).to_string(index=False))
    n = len(D); idx = rng.integers(0, n, (3000, n))
    print("Paired-bootstrap MAE differences (3000 resamples):")
    for base in ["fixed2.5", "fixed1.75", "roll5_mean"]:
        for other in ["roll5_mean","roll5_median","allprior_mean","fixed1.5","fixed1.75","fixed2.0","fixed2.5","nfelo_pergame"]:
            if other == base: continue
            dd = np.abs(store[other]) - np.abs(store[base]); bs = dd[idx].mean(axis=1)
            print("  MAE(%-14s) - MAE(%-10s) = %+.3f  95%% CI [%+.3f, %+.3f]" % (other, base, dd.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)))
    # ATS: does leaning home (bigger k) pick home more and lose?  home-pick share for each k
    for k in ["fixed1.5","fixed2.0","fixed2.5","roll5_mean"]:
        pred = -(D.rating_dif.values + {"fixed1.5":1.5,"fixed2.0":2.0,"fixed2.5":2.5}.get(k, 0)) if k!="roll5_mean" else -(D.margin.values - store[k])
        print("  %-10s share of picks on home vs market = %.3f" % (k, (pred < D.mkt_spread.values).mean()))

# ---------- market-vs-mean check: is a +0.6 home residual real? ----------
t = m[m.season>=2022]
print("\n2022-25 market residual (margin + mkt_spread): mean %+.2f (se %.2f, t=%.2f), median %+.2f, home cover %.3f, home ML win %.3f" %
      (t.spread_err_mkt.mean(), t.spread_err_mkt.std()/np.sqrt(len(t)), t.spread_err_mkt.mean()/(t.spread_err_mkt.std()/np.sqrt(len(t))), t.spread_err_mkt.median(), (t.spread_err_mkt>0).sum()/(t.spread_err_mkt!=0).sum(), (t.margin>0).mean()))
print("skew of market residual 2022-25: %.2f ; 2009-19: %.2f" % (stats.skew(t.spread_err_mkt), stats.skew(m[m.season<=2019].spread_err_mkt)))
