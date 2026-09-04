"""CRITIC extras: (a) which reconstruction of nfelo_dif_base matches, by season (what is hfa_mod?);
(b) dome/outdoor HFA split Welch test; (c) leave-one-season-out robustness of MAE(2.5)-MAE(1.75) and
the OOS engine-proxy divisional totals bias gap with its se."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from kit import merged, load_nfelo
pd.set_option("display.width", 220)
n_ = load_nfelo(); n_ = n_.dropna(subset=["nfelo_dif_base"]).copy(); n_["tz"] = n_.home_time_advantage_mod.fillna(0)
core = (n_.starting_nfelo_home - n_.starting_nfelo_away + n_.home_net_qb_mod.fillna(0) + n_.home_net_bye_mod.fillna(0) + n_.div_game_mod.fillna(0) + n_.dif_surface_mod.fillna(0))
R = pd.DataFrame({"season": n_.season,
                  "base+tz": ((n_.nfelo_dif_base - core - n_.hfa_base_mod - n_.tz).abs() < 1),
                  "hfa_mod": ((n_.nfelo_dif_base - core - n_.hfa_mod).abs() < 1),
                  "hfa_mod+tz": ((n_.nfelo_dif_base - core - n_.hfa_mod - n_.tz).abs() < 1)})
print("(a) share of games where nfelo_dif_base == core + X (|res|<1 Elo), by season:"); print(R.groupby("season").mean().round(2).T.to_string())
bad = n_[(n_.nfelo_dif_base - core - n_.hfa_base_mod - n_.tz).abs() >= 1]
print("    games not matching base+tz: n=%d; of these, share with nonzero bye mods = %.2f; residual mean %.1f" % (len(bad), (bad.home_net_bye_mod.fillna(0)!=0).mean(), (bad.nfelo_dif_base - core - bad.hfa_base_mod - bad.tz).mean()))
print("    => nfelo's own line uses hfa_base_mod + time-zone mod; the hfa_mod column (sd %.0f Elo) is NOT what nfelo adds to its line." % n_.hfa_mod.std())
# what is hfa_mod? regress hfa_mod on hfa_base_mod, tz, and home-team dummies (team-specific component?)
import statsmodels.formula.api as smf
d = n_[n_.season>=2021].copy()
o = smf.ols("hfa_mod ~ hfa_base_mod + tz + C(home)", data=d).fit()
print("    2021-25: hfa_mod ~ hfa_base_mod + tz + home-team FE: R2=%.3f (vs %.3f without team FE); b_tz=%.2f" % (o.rsquared, smf.ols("hfa_mod ~ hfa_base_mod + tz", data=d).fit().rsquared, o.params["tz"]))

# (b) dome / outdoor split
m = merged(); m = m[m.game_type.eq("REG") & ~m.neutral].copy()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m = m.dropna(subset=["rating_dif","mkt_spread"]).copy(); m["r"] = m.margin - m.rating_dif
for lab, d in [("2021-25", m[m.season>=2021]), ("2009-25 ex2020", m[m.season!=2020])]:
    a, b = d[d.is_dome].r, d[~d.is_dome].r
    print("\n(b) %s dome/closed HFA %.2f (n=%d) vs outdoor/open %.2f (n=%d): diff %+.2f, Welch p=%.3f; median %.2f vs %.2f" % (lab, a.mean(), len(a), b.mean(), len(b), a.mean()-b.mean(), stats.ttest_ind(a, b, equal_var=False).pvalue, a.median(), b.median()))

# (c) leave-one-season-out for MAE(2.5) - MAE(1.75), 2015-25 ex 2020 and 2022-25
x = m[m.season!=2020]
for lo in [2015, 2022]:
    t = x[x.season>=lo]; dd = np.abs(t.r - 2.5) - np.abs(t.r - 1.75)
    print("\n(c) MAE(2.5)-MAE(1.75) %d-25: all %+.3f ; leave-one-season-out:" % (lo, dd.mean()), {s: round(dd[t.season!=s].mean(), 3) for s in sorted(t.season.unique())})
    print("    per-season:", {s: round(dd[t.season==s].mean(), 3) for s in sorted(t.season.unique())})
# engine-proxy div bias gap OOS
mm = merged(); mm = mm[mm.game_type.eq("REG") & ~mm.neutral].dropna(subset=["mkt_total","starting_nfelo_home"]).copy()
mm["rating_sum"] = mm.home_pts_vs_avg + mm.away_pts_vs_avg; mm["div"] = mm.div_game.astype(int)
recs = []
for s in [s for s in range(2014, 2026) if s != 2020]:
    f = mm[(mm.season<s)&(mm.season!=2020)]; te = mm[mm.season==s].copy(); last = f[f.season==max(f.season)]
    te["e"] = te.total_pts - (last.total_pts.mean() + 0.35*te.rating_sum); recs.append(te)
T = pd.concat(recs); a, b = T[T["div"]==1].e, T[T["div"]==0].e
print("\n    engine-proxy (prior-season mean + 0.35*rating_sum) OOS 2014-25 ex2020: bias div %+.2f (n=%d) vs non-div %+.2f (n=%d); gap %+.2f (se %.2f, Welch p=%.3f)" %
      (a.mean(), len(a), b.mean(), len(b), a.mean()-b.mean(), np.sqrt(a.var()/len(a)+b.var()/len(b)), stats.ttest_ind(a, b, equal_var=False).pvalue))
am, bm = T[T["div"]==1].total_err_mkt, T[T["div"]==0].total_err_mkt
print("    market OOS same games: bias div %+.2f vs non-div %+.2f; gap %+.2f (se %.2f, p=%.3f)" % (am.mean(), bm.mean(), am.mean()-bm.mean(), np.sqrt(am.var()/len(am)+bm.var()/len(bm)), stats.ttest_ind(am, bm, equal_var=False).pvalue))
