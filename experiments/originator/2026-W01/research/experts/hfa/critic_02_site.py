"""CRITIC HFA-2: (1) what is inside nfelo hfa_mod vs hfa_base_mod vs home_time_advantage_mod (the 'double count' claim);
(2) rolling-origin test of team-specific HFA (EB and raw) and of nfelo per-game HFA over 2014-2025 pooled (n~2900, more power
than the 2022-25 window alone); (3) odd/even-season split-half replication + one-way ANOVA for any team heterogeneity;
(4) Denver season-by-season."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from kit import merged, load_nfelo
pd.set_option("display.width", 220)
rng = np.random.default_rng(11)

# ---------- (1) composition ----------
n_ = load_nfelo(); n_ = n_.dropna(subset=["nfelo_dif_base"]).copy(); n_["tz"] = n_.home_time_advantage_mod.fillna(0)
print("(1) nfelo hfa_mod composition (Elo units), n=%d" % len(n_))
o = sm.OLS(n_.hfa_mod, sm.add_constant(n_[["hfa_base_mod","tz"]])).fit()
print("  hfa_mod ~ hfa_base_mod + tz: b_base=%.3f  b_tz=%.3f  R2=%.3f" % (o.params["hfa_base_mod"], o.params["tz"], o.rsquared))
print("  hfa_mod == hfa_base_mod in %.1f%% of games; (hfa_mod - hfa_base_mod): mean %.2f sd %.2f Elo" % ((n_.hfa_mod==n_.hfa_base_mod).mean()*100, (n_.hfa_mod-n_.hfa_base_mod).mean(), (n_.hfa_mod-n_.hfa_base_mod).std()))
print("  corr(hfa_mod - hfa_base_mod, tz) = %.2f ; corr(hfa_base_mod, tz) = %.2f" % (np.corrcoef(n_.hfa_mod-n_.hfa_base_mod, n_.tz)[0,1], np.corrcoef(n_.hfa_base_mod, n_.tz)[0,1]))
core = (n_.starting_nfelo_home - n_.starting_nfelo_away + n_.home_net_qb_mod.fillna(0) + n_.home_net_bye_mod.fillna(0)
        + n_.div_game_mod.fillna(0) + n_.dif_surface_mod.fillna(0))
for nm, comp in [("core + hfa_mod", core + n_.hfa_mod), ("core + hfa_mod + tz", core + n_.hfa_mod + n_.tz),
                 ("core + hfa_base_mod", core + n_.hfa_base_mod), ("core + hfa_base_mod + tz", core + n_.hfa_base_mod + n_.tz)]:
    res = n_.nfelo_dif_base - comp
    print("  nfelo_dif_base - (%-24s): mean %+.2f  sd %.2f  |res|<1 share %.2f" % (nm, res.mean(), res.std(), (res.abs()<1).mean()))
print("  by-season means (Elo): base / hfa_mod / tz / (hfa_mod-base):")
print(n_.groupby("season").apply(lambda d: pd.Series(dict(hfa_base=d.hfa_base_mod.mean(), hfa_mod=d.hfa_mod.mean(), tz=d.tz.mean(), diff=(d.hfa_mod-d.hfa_base_mod).mean(),
      sd_base=d.hfa_base_mod.std(), sd_hfa=d.hfa_mod.std()))).round(1).to_string())
# how does hfa_mod - hfa_base_mod relate to tz within 2022-25: table by tz sign
d = n_[n_.season>=2022]
print("  2022-25 mean(hfa_mod - hfa_base_mod) by sign(tz):", d.groupby(np.sign(d.tz)).apply(lambda z: (z.hfa_mod-z.hfa_base_mod).mean()).round(1).to_dict())

# ---------- (2) rolling-origin team HFA ----------
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy(); m["r"] = m.margin - m.rating_dif
m["nfelo_hfa"] = m.hfa_pts; m["nfelo_base"] = m.hfa_mod/25
print("\n(2) rolling-origin team HFA, test seasons 2014-2025 ex 2020, fit = all prior seasons ex 2020")
def eb_table(f):
    f = f.copy(); f["x"] = f.r - f.groupby("season").r.transform("mean")
    g = f.groupby("home").x.agg(["mean","count","std"]); g["se2"] = g["std"]**2/g["count"]
    tau2 = max(0.0, g["mean"].var(ddof=1) - g["se2"].mean()); g["eb"] = tau2/(tau2+g["se2"])*g["mean"]
    return g, tau2
rows, recs = [], []
for s in [s for s in range(2014, 2026) if s != 2020]:
    fitd = m[(m.season<s)&(m.season!=2020)]; te = m[m.season==s].copy()
    G, tau2 = eb_table(fitd); ps = sorted(set(fitd.season)); K = fitd[fitd.season.isin(ps[-5:])].r.mean()
    te["eb"] = te.home.map(G.eb).fillna(0); te["raw"] = te.home.map(G["mean"]).fillna(0); te["K"] = K
    last = fitd[fitd.season==ps[-1]]
    te["nf_ex"] = te.nfelo_hfa - last.nfelo_hfa.mean(); te["nfb_ex"] = te.nfelo_base - last.nfelo_base.mean()
    recs.append(te); rows.append(dict(season=s, n_fit=len(fitd), tau=np.sqrt(tau2), K=K, sd_eb=te.eb.std(), sd_nfelo_ex=te.nf_ex.std()))
print(pd.DataFrame(rows).round(2).to_string(index=False))
T = pd.concat(recs); T["y"] = T.r - T.K
print("Pooled OOS n=%d. Regress realized excess (r - K) on predicted excess [slope 1 = fully predictive]:" % len(T))
for nm, col in [("EB fit excess","eb"),("raw fit excess","raw"),("nfelo hfa_pts excess","nf_ex"),("nfelo hfa_mod excess","nfb_ex")]:
    o = sm.OLS(T.y, sm.add_constant(T[col])).fit(cov_type="HC1")
    o2 = smf.ols(f"y ~ {col} + C(season)", data=T).fit(cov_type="HC1")
    print("  %-22s slope=%.2f (se %.2f, p=%.3f) | with season FE slope=%.2f (se %.2f, p=%.3f) | sd(x)=%.2f" %
          (nm, o.params.iloc[1], o.bse.iloc[1], o.pvalues.iloc[1], o2.params[col], o2.bse[col], o2.pvalues[col], T[col].std()))
for lo, hi in [(2014,2019),(2021,2025)]:
    t = T[(T.season>=lo)&(T.season<=hi)]
    for col in ["eb","nf_ex"]:
        o = sm.OLS(t.y, sm.add_constant(t[col])).fit(cov_type="HC1"); print("  %d-%d  y ~ %-6s slope=%.2f (se %.2f, p=%.3f) n=%d" % (lo,hi,col,o.params.iloc[1],o.bse.iloc[1],o.pvalues.iloc[1],len(t)))
e0 = (T.margin - T.rating_dif - T.K).values; idx = rng.integers(0, len(T), (3000, len(T)))
print("Pooled OOS MAE (const K = %.3f):" % np.abs(e0).mean())
for nm, e in [("const+EB", e0 - T.eb.values), ("const+0.5EB", e0 - 0.5*T.eb.values), ("const+raw", e0 - T.raw.values),
              ("nfelo hfa_pts", (T.margin - T.rating_dif - T.nfelo_hfa).values), ("nfelo hfa_mod", (T.margin - T.rating_dif - T.nfelo_base).values)]:
    dd = np.abs(e) - np.abs(e0); bs = dd[idx].mean(axis=1)
    print("  MAE(%-13s) - MAE(const K) = %+.3f  95%% CI [%+.3f, %+.3f]" % (nm, dd.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)))

# ---------- (3) split-half and ANOVA ----------
x = m[m.season!=2020].copy(); x["ex"] = x.r - x.groupby("season").r.transform("mean")
odd = x[x.season%2==1].groupby("home").ex.agg(["mean","count"]); even = x[x.season%2==0].groupby("home").ex.agg(["mean","count"])
j = odd.join(even, lsuffix="_odd", rsuffix="_even").dropna(); j = j[(j.count_odd>=25)&(j.count_even>=25)]
r_, p_ = stats.pearsonr(j.mean_odd, j.mean_even)
print("\n(3) odd vs even seasons 2009-25 ex 2020: team HFA excess corr r=%.2f (p=%.3f, %d teams)" % (r_, p_, len(j)))
a = x[x.season<=2014].groupby("home").ex.agg(["mean","count"]); b = x[x.season>=2015].groupby("home").ex.agg(["mean","count"])
j2 = a.join(b, lsuffix="_a", rsuffix="_b").dropna(); j2 = j2[(j2.count_a>=25)&(j2.count_b>=25)]
print("   2009-14 vs 2015-25: r=%.2f (p=%.3f, %d teams)" % (*stats.pearsonr(j2.mean_a, j2.mean_b), len(j2)))
groups = [g.ex.values for _, g in x.groupby("home") if len(g) >= 25]
F, pF = stats.f_oneway(*groups)
kw = stats.kruskal(*groups)
print("   one-way ANOVA of game residual by home team (2009-25 ex 2020): F=%.2f p=%.3f ; Kruskal p=%.3f" % (F, pF, kw.pvalue))
# same with market residual (does the market misprice teams' HFA?)
x["mex"] = x.spread_err_mkt - x.groupby("season").spread_err_mkt.transform("mean")
groups = [g.mex.values for _, g in x.groupby("home") if len(g) >= 25]
print("   one-way ANOVA of MARKET residual by home team: F=%.2f p=%.3f" % stats.f_oneway(*groups))

# ---------- (4) Denver ----------
den = x[x.home.eq("DEN")].groupby("season").agg(n=("ex","size"), excess=("ex","mean"), mkt_resid=("spread_err_mkt","mean"))
print("\n(4) Denver excess over league by season:"); print(den.round(2).T.to_string())
print("   seasons with positive excess: %d of %d ; mean %.2f (se %.2f)" % ((den.excess>0).sum(), len(den), x[x.home.eq("DEN")].ex.mean(), x[x.home.eq("DEN")].ex.std()/np.sqrt(len(x[x.home.eq("DEN")]))))
print("   DEN market residual 2009-25 ex2020: mean %+.2f (se %.2f), home cover %.3f (n=%d)" % (x[x.home.eq("DEN")].spread_err_mkt.mean(), x[x.home.eq("DEN")].spread_err_mkt.std()/np.sqrt(len(x[x.home.eq("DEN")])), (x[x.home.eq("DEN")].spread_err_mkt>0).sum()/(x[x.home.eq("DEN")].spread_err_mkt!=0).sum(), len(x[x.home.eq("DEN")])))
