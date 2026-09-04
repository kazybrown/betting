"""CRITIC HFA-3: slot HFA. (a) pooled 2009-2025 ex 2020 with season FE; (b) permutation placebo for the in-sample SNF effect
(max |t| across slot dummies when slot labels are shuffled within season); (c) odd/even split-half; (d) rolling-origin
pooled 2014-2025; (e) Week 1 pooled, incl. market residual and win rate."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from kit import merged
pd.set_option("display.width", 220)
rng = np.random.default_rng(5)
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy(); m["r"] = m.margin - m.rating_dif
m["hour"] = m.gametime.str.slice(0,2).astype(int) + m.gametime.str.slice(3,5).astype(int)/60
def slot(r):
    if r.weekday == "Sunday": return "SUN_early" if r.hour < 15 else ("SUN_late" if r.hour < 19 else "SNF")
    return {"Monday":"MNF","Thursday":"TNF","Saturday":"SAT"}.get(r.weekday, "OTHER")
m["slot"] = m.apply(slot, axis=1); m["wk1"] = m.week.eq(1).astype(int)
cats = ["SUN_early","SUN_late","SNF","MNF","TNF","SAT","OTHER"]
m["slotc"] = pd.Categorical(m.slot, categories=cats)
x = m[m.season!=2020].copy()
print("(a) pooled 2009-2025 ex 2020 (n=%d): r ~ slot + wk1 + season FE (ref SUN_early), HC1" % len(x))
mod = smf.ols("r ~ C(slotc) + wk1 + C(season)", data=x).fit(cov_type="HC1")
for c in cats[1:]:
    k = f"C(slotc)[T.{c}]"; print("  %-9s %+.2f (se %.2f, p=%.3f)" % (c, mod.params[k], mod.bse[k], mod.pvalues[k]))
print("  wk1       %+.2f (se %.2f, p=%.3f)" % (mod.params["wk1"], mod.bse["wk1"], mod.pvalues["wk1"]))
modm = smf.ols("spread_err_mkt ~ C(slotc) + wk1 + C(season)", data=x).fit(cov_type="HC1")
print("  market residual: " + ", ".join("%s %+.2f (p=%.2f)" % (c, modm.params[f"C(slotc)[T.{c}]"], modm.pvalues[f"C(slotc)[T.{c}]"]) for c in cats[1:]) + ", wk1 %+.2f (p=%.2f)" % (modm.params["wk1"], modm.pvalues["wk1"]))

# (b) permutation placebo in the expert's fit window
fit = m[(m.season<=2021)&(m.season!=2020)].copy()
def max_t(df):
    mo = smf.ols("r ~ C(slotc) + wk1 + C(season)", data=df).fit(cov_type="HC1")
    t = (mo.params/mo.bse).filter(like="slotc"); t = t.drop("C(slotc)[T.OTHER]", errors="ignore")
    return t.abs().max()
obs = max_t(fit); B = 400; perm = np.empty(B)
for b in range(B):
    f = fit.copy(); f["slotc"] = pd.Categorical(f.groupby("season").slot.transform(lambda s: rng.permutation(s.values)), categories=cats); perm[b] = max_t(f)
print("\n(b) fit window 2009-21: observed max |t| over 5 slot dummies = %.2f (SNF); permutation p (labels shuffled within season, B=%d) = %.3f" % (obs, B, (perm>=obs).mean()))

# (c) odd/even split
def dev(df):
    ref = df[df.slot.isin(["SUN_early","SUN_late"])].r.mean()
    return pd.Series({s: df[df.slot==s].r.mean()-ref for s in ["SNF","MNF","TNF","SAT"]} | {"WK1": df[df.wk1==1].r.mean()-ref})
odd, even = dev(x[x.season%2==1]), dev(x[x.season%2==0])
print("\n(c) slot deviation from Sun-aft: odd seasons vs even seasons"); print(pd.DataFrame({"odd":odd, "even":even}).round(2).T.to_string())

# (d) rolling-origin pooled
recs = []
for s in [s for s in range(2014, 2026) if s != 2020]:
    f = m[(m.season<s)&(m.season!=2020)]; te = m[m.season==s].copy()
    mo = smf.ols("r ~ C(slotc) + wk1 + C(season)", data=f).fit()
    co = {c: mo.params.get(f"C(slotc)[T.{c}]", 0.0) for c in cats}; co["SUN_early"] = 0.0
    adj_f = f.slot.map(co) + mo.params["wk1"]*f.wk1
    te["adj"] = te.slot.map(co) + mo.params["wk1"]*te.wk1 - adj_f.mean()
    ps = sorted(set(f.season)); te["K"] = f[f.season.isin(ps[-5:])].r.mean(); recs.append(te)
T = pd.concat(recs); e0 = T.r - T.K; e1 = e0 - T.adj
o = sm.OLS(e0, sm.add_constant(T.adj)).fit(cov_type="HC1")
print("\n(d) rolling-origin 2014-25 ex 2020 (n=%d): residual ~ fitted slot adj slope=%.2f (se %.2f, p=%.3f); MAE const=%.3f, const+slot=%.3f" % (len(T), o.params.iloc[1], o.bse.iloc[1], o.pvalues.iloc[1], e0.abs().mean(), e1.abs().mean()))
om = sm.OLS(T.spread_err_mkt, sm.add_constant(T.adj)).fit(cov_type="HC1")
print("    market residual ~ fitted slot adj slope=%.2f (se %.2f, p=%.3f)" % (om.params.iloc[1], om.bse.iloc[1], om.pvalues.iloc[1]))

# (e) Week 1
w = x[x.wk1==1]; nw = x[x.wk1==0]
print("\n(e) Week 1 (n=%d) vs weeks 2+ (n=%d), 2009-25 ex 2020: rating-HFA %.2f vs %.2f (Welch p=%.3f); market resid %+.2f vs %+.2f (p=%.3f); home cover %.3f vs %.3f; home win %.3f vs %.3f" %
      (len(w), len(nw), w.r.mean(), nw.r.mean(), stats.ttest_ind(w.r, nw.r, equal_var=False).pvalue, w.spread_err_mkt.mean(), nw.spread_err_mkt.mean(),
       stats.ttest_ind(w.spread_err_mkt, nw.spread_err_mkt, equal_var=False).pvalue, (w.spread_err_mkt>0).sum()/(w.spread_err_mkt!=0).sum(), (nw.spread_err_mkt>0).sum()/(nw.spread_err_mkt!=0).sum(), (w.margin>0).mean(), (nw.margin>0).mean()))
print("    Week 1 rating-HFA by season:", x[x.wk1==1].groupby("season").r.mean().round(1).to_dict())
print("    seasons with Week-1 HFA below that season's overall HFA: %d of %d" % (sum(x[(x.season==s)&(x.wk1==1)].r.mean() < x[x.season==s].r.mean() for s in sorted(x.season.unique())), x.season.nunique()))
