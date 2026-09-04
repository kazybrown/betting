"""critic_02_T1a_power_placebo.py - CRITIC of T1a (disagreement -> market |err|, REJECTED by the expert).
A null is only as good as its power. Here:
 (1) pooled n=4310 slope with 95% CI and the implied bound on the LOW/HIGH variance ratio;
 (2) per-season slope of |e_mkt| on D_base (17 seasons) -> sign test; same for e_mkt^2;
 (3) Brown-Forsythe (Levene, median-centred) test of e_mkt across D bands, full sample;
 (4) top-decile D vs rest variance ratio with bootstrap CI;
 (5) placebo: D drawn at random from its own distribution -> same tests (should be null);
 (6) season fixed effects (era confound) and playoff exclusion;
 (7) the same with an independent engine (qbelo) and with the engine mean;
 (8) totals: is mkt_total a variance driver OOS? (the one marginal driver the expert saw)
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_02_T1a_power_placebo.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

pd.set_option("display.width", 200)
m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
rng = np.random.default_rng(42)


def slope(y, x, X_extra=None):
    X = np.asarray(x, float)[:, None] if X_extra is None else np.column_stack([np.asarray(x, float), X_extra])
    r = sm.OLS(np.asarray(y, float), sm.add_constant(X)).fit(cov_type="HC1")
    return r.params[1], r.bse[1], r.pvalues[1], r.conf_int()[1]


print("(1) POOLED slope of market |err| on D_base, n=%d" % len(m))
b, se, p, ci = slope(m.ae_mkt, m.D_base)
print(f"  |e_mkt| ~ D_base: slope {b:+.3f} se {se:.3f} p={p:.3f} 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
b2, se2, p2, ci2 = slope(m.e_mkt**2, m.D_base)
print(f"  e_mkt^2 ~ D_base: slope {b2:+.2f} se {se2:.2f} p={p2:.3f} 95% CI [{ci2[0]:+.2f}, {ci2[1]:+.2f}]  (variance per point of D)")
mean_var = (m.e_mkt**2).mean(); D_hi, D_lo = m.loc[m.D_base >= 3, "D_base"].mean(), m.loc[m.D_base < 1.5, "D_base"].mean()
print(f"  implied LOW(D>=3, mean D {D_hi:.2f}) / HIGH(D<1.5, mean D {D_lo:.2f}) variance ratio: point {1 + b2*(D_hi-D_lo)/mean_var:.3f}, upper 95% {1 + ci2[1]*(D_hi-D_lo)/mean_var:.3f}, lower {1 + ci2[0]*(D_hi-D_lo)/mean_var:.3f}")
print(f"  -> in sigma terms the LOW band could be at most sqrt(upper)= {np.sqrt(1 + ci2[1]*(D_hi-D_lo)/mean_var):.3f}x the HIGH band sigma (~{12.5*(np.sqrt(1 + ci2[1]*(D_hi-D_lo)/mean_var)-1):.2f} pts on a 12.5 base)")

print("\n(2) PER-SEASON slopes (sign test across 17 seasons)")
rows = []
for s, d in m.groupby("season"):
    b1 = slope(d.ae_mkt, d.D_base)[0]; b2s = slope(d.e_mkt**2, d.D_base)[0]
    rows.append(dict(season=s, n=len(d), slope_abs=b1, slope_sq=b2s))
t = pd.DataFrame(rows).set_index("season")
print(t.round(2).T.to_string())
for c in ["slope_abs", "slope_sq"]:
    k = int((t[c] > 0).sum()); print(f"  {c}: positive in {k}/{len(t)} seasons (binom p={stats.binomtest(k, len(t)).pvalue:.3f}); mean {t[c].mean():+.3f}, t-test vs 0 p={stats.ttest_1samp(t[c], 0).pvalue:.3f}")

print("\n(3) Brown-Forsythe test of e_mkt across D_base bands (full sample / fit / test)")
for lab, d in [("full", m), ("fit", fit), ("test", test)]:
    bands = [d.e_mkt[d.D_base < 1.5], d.e_mkt[(d.D_base >= 1.5) & (d.D_base < 3)], d.e_mkt[d.D_base >= 3]]
    W, p = stats.levene(*bands, center="median")
    print(f"  {lab}: n={[len(b) for b in bands]} sd={[round(b.std(),2) for b in bands]} Brown-Forsythe W={W:.2f} p={p:.3f}")

print("\n(4) top-decile D_base vs rest: variance ratio with bootstrap CI (full sample)")
cut = m.D_base.quantile(.9); hi, lo = m.e_mkt[m.D_base >= cut].values, m.e_mkt[m.D_base < cut].values
ratio = (hi**2).mean() / (lo**2).mean()
bs = [(rng.choice(hi, len(hi))**2).mean() / (rng.choice(lo, len(lo))**2).mean() for _ in range(2000)]
print(f"  D_base >= {cut:.2f}: n={len(hi)} rmse {np.sqrt((hi**2).mean()):.2f} vs rest rmse {np.sqrt((lo**2).mean()):.2f} | variance ratio {ratio:.3f} 95% CI [{np.quantile(bs,.025):.3f}, {np.quantile(bs,.975):.3f}]")

print("\n(5) PLACEBO: D permuted across games (breaks any real link) - 200 reps, distribution of the pooled slope")
sl = []
for _ in range(200):
    Dp = rng.permutation(m.D_base.values); sl.append(slope(m.ae_mkt, Dp)[0])
sl = np.array(sl)
print(f"  placebo slope mean {sl.mean():+.3f} sd {sl.std():.3f} | real pooled slope {b:+.3f} -> percentile of real among placebos {100*(sl < b).mean():.0f}%")

print("\n(6) confounds: season fixed effects; REG only; abs_mkt + mkt_total + dome controls")
Xs = pd.get_dummies(m.season, drop_first=True).astype(float).values
b6, se6, p6, ci6 = slope(m.ae_mkt, m.D_base, Xs); print(f"  with season FE: slope {b6:+.3f} se {se6:.3f} p={p6:.3f}")
r = m[m.playoff == 0]; b7, se7, p7, _ = slope(r.ae_mkt, r.D_base, pd.get_dummies(r.season, drop_first=True).astype(float).values); print(f"  REG only + season FE (n={len(r)}): slope {b7:+.3f} se {se7:.3f} p={p7:.3f}")
b8, se8, p8, _ = slope(m.ae_mkt, m.D_base, np.column_stack([Xs, m[["abs_mkt", "mkt_total", "dome"]].astype(float).values])); print(f"  + abs_mkt, mkt_total, dome: slope {b8:+.3f} se {se8:.3f} p={p8:.3f}")

print("\n(7) independent engine (538 qbelo) and engine mean as the disagreement source")
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
s = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
nn = n.iloc[: len(s)].reset_index(drop=True); s["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
q = m.merge(s[["gid", "qbelo_home_line_close_rounded"]].rename(columns={"qbelo_home_line_close_rounded": "qbelo"}), on="gid", how="inner").dropna(subset=["qbelo"])
q["D_q"] = (q.qbelo - q.mkt).abs(); q["mean2"] = (q.qbelo + q.nfelo_b) / 2; q["D_m2"] = (q.mean2 - q.mkt).abs(); q["sd2"] = (q.qbelo - q.nfelo_b).abs() / np.sqrt(2)
for col in ["D_q", "D_m2", "sd2"]:
    b9, se9, p9, ci9 = slope(q.ae_mkt, q[col]); bt, sbt, pt, _ = slope(q[q.era == "test"].ae_mkt, q[q.era == "test"][col])
    print(f"  |e_mkt| ~ {col:5s}: pooled n={len(q)} slope {b9:+.3f} se {se9:.3f} p={p9:.3f} CI [{ci9[0]:+.3f},{ci9[1]:+.3f}] | test-only slope {bt:+.3f} p={pt:.3f}")

print("\n(8) TOTALS: mkt_total as a variance driver - rolling-origin (fit through s-1, test s) 2016-25")
rows = []
for s_ in range(2016, 2026):
    f = m[m.season < s_]; t = m[m.season == s_]
    A = sm.OLS(f.ae_tot.values, sm.add_constant(f.mkt_total.values)).fit()
    pred = A.predict(sm.add_constant(t.mkt_total.values)); rho = stats.spearmanr(pred, t.ae_tot)[0]
    rows.append(dict(season=s_, slope_fit=A.params[1], rho_oos=rho, n=len(t)))
t8 = pd.DataFrame(rows).set_index("season"); print(t8.round(3).T.to_string())
print(f"  mean OOS Spearman {t8.rho_oos.mean():+.3f}; positive in {(t8.rho_oos>0).sum()}/10 seasons | pooled |e_tot| ~ mkt_total slope {slope(m.ae_tot, m.mkt_total)[0]:+.3f} p={slope(m.ae_tot, m.mkt_total)[2]:.3f}")
lo_, hi_ = m[m.mkt_total <= 41], m[m.mkt_total >= 49]
print(f"  total<=41 (n={len(lo_)}): total-error RMSE {np.sqrt((lo_.e_tot**2).mean()):.2f} | total>=49 (n={len(hi_)}): {np.sqrt((hi_.e_tot**2).mean()):.2f} | Brown-Forsythe p={stats.levene(lo_.e_tot, hi_.e_tot, center='median')[1]:.3f}")
