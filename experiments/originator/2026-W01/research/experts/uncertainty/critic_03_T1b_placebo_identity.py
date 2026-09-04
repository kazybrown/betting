"""critic_03_T1b_placebo_identity.py - CRITIC of T1b (disagreement -> MODEL |err|, SUPPORTED by the expert).
If e_model = e_mkt + d is all there is, then a PLACEBO model = market + pure noise (noise matched to the
D_base distribution) must reproduce the expert's slope, Spearman and tercile pattern exactly. If it does,
T1b contains no information about nfelo: it is the identity. What is NOT tautological is the covariance
term 2*cov(e_mkt, d): negative = the model has edge (its distance is partly signal), zero = none,
positive = the model is anti-informative. That is the number to report.
 (1) placebo replication (500 reps);
 (2) the excess beyond the identity, by band and era, for nfelo_b, qbelo, and the engine mean;
 (3) robustness of the expert's OOS slope: winsorised D, drop D>4.5, Huber regression, rolling-origin;
 (4) the same slope for the market's error (control) inside the same reps.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_03_T1b_placebo_identity.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, tercile_labels

pd.set_option("display.width", 200)
m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
rng = np.random.default_rng(7)


def slope(y, x):
    r = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit(cov_type="HC1")
    return r.params[1], r.bse[1], r.pvalues[1]


print("(1) PLACEBO model = market close + signed noise with the SAME |d| distribution as nfelo_b (|d| resampled from D_base, sign random)")
real_b, real_se, real_p = slope(test.ae_nb, test.D_base); real_rho = stats.spearmanr(test.D_base, test.ae_nb)[0]
lab, edges = tercile_labels(test.D_base.values, ref=fit.D_base.values)
real_t = [np.sqrt((test.e_nb[lab == k] ** 2).mean()) for k in ["T1 low", "T2 mid", "T3 high"]]
print(f"  REAL nfelo_b (test): slope {real_b:+.3f} (se {real_se:.3f}, p={real_p:.3f}) Spearman {real_rho:+.3f} | tercile RMSE {np.round(real_t,2)}")
sl, rh, t3 = [], [], []
for _ in range(500):
    d = rng.choice(test.D_base.values, len(test)) * rng.choice([-1, 1], len(test))
    e_f = test.e_mkt.values + d; D_f = np.abs(d)
    sl.append(slope(np.abs(e_f), D_f)[0]); rh.append(stats.spearmanr(D_f, np.abs(e_f))[0])
    lf, _ = tercile_labels(D_f, edges=edges); t3.append([np.sqrt((e_f[lf == k] ** 2).mean()) for k in ["T1 low", "T2 mid", "T3 high"]])
sl, rh, t3 = np.array(sl), np.array(rh), np.array(t3)
print(f"  PLACEBO (500 reps): slope mean {sl.mean():+.3f} [2.5-97.5%: {np.quantile(sl,.025):+.3f}, {np.quantile(sl,.975):+.3f}] | Spearman mean {rh.mean():+.3f} | tercile RMSE mean {np.round(t3.mean(0),2)}")
print(f"  real slope percentile among placebos: {100*(sl < real_b).mean():.0f}% ; real Spearman percentile: {100*(rh < real_rho).mean():.0f}%")
print("  -> if the real numbers sit inside the placebo band, T1b is the identity and says nothing about nfelo.")

print("\n(2) EXCESS beyond the identity: RMSE_model^2 - RMSE_mkt^2 - mean(d^2) = 2*cov(e_mkt, d)  [negative = model has edge]")
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
s = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
nn = n.iloc[: len(s)].reset_index(drop=True); s["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
q = m.merge(s[["gid", "qbelo_home_line_close_rounded"]].rename(columns={"qbelo_home_line_close_rounded": "qbelo"}), on="gid", how="inner").dropna(subset=["qbelo"]).copy()
q["mean2"] = (q.qbelo + q.nfelo_b) / 2


def excess_table(df, model, label):
    d = df[model] - df.mkt; e_m = df.margin + df[model]
    rows = []
    for era in ["fit", "test"]:
        for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3, "MED"), (3, 99, "LOW"), (0, 99, "ALL")]:
            mk = (df.era == era) & (d.abs() >= lo) & (d.abs() < hi)
            em, ek, dd = e_m[mk], df.e_mkt[mk], d[mk]
            v_m, v_k, d2 = (em**2).mean(), (ek**2).mean(), (dd**2).mean()
            cov2 = 2 * np.mean(ek * dd)
            # bootstrap CI for 2cov
            idx = rng.integers(0, mk.sum(), (2000, mk.sum())); bs = 2 * (ek.values[idx] * dd.values[idx]).mean(1)
            rows.append(dict(era=era, band=name, n=int(mk.sum()), rmse_mkt=np.sqrt(v_k), rmse_model=np.sqrt(v_m), identity_pred=np.sqrt(v_k + d2), two_cov=cov2, ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975), corr=np.corrcoef(ek, dd)[0, 1],
                             w_implied=-np.mean(ek * dd) / d2))
    t = pd.DataFrame(rows); print(f"\n  {label}"); print(t.round(3).to_string(index=False))
    return t


excess_table(m, "nfelo_b", "nfelo_b vs nflverse close (expert's frame)")
excess_table(q, "qbelo", "538 qbelo (independent engine) vs close")
excess_table(q, "mean2", "engine mean (nfelo_b, qbelo) vs close")
print("  w_implied = -cov(e_mkt,d)/E[d^2] is the no-intercept blend weight of T3 inside each band (0 = market right).")

print("\n(3) ROBUSTNESS of the expert's OOS slope +0.364 (|e_nb| ~ D_base, test era)")
t = test.copy()
print(f"  winsorise D at 4.5: slope {slope(t.ae_nb, t.D_base.clip(upper=4.5))[0]:+.3f} p={slope(t.ae_nb, t.D_base.clip(upper=4.5))[2]:.3f}")
u = t[t.D_base < 4.5]; print(f"  drop D>=4.5 (n={len(u)}): slope {slope(u.ae_nb, u.D_base)[0]:+.3f} p={slope(u.ae_nb, u.D_base)[2]:.3f}")
hub = sm.RLM(t.ae_nb.values, sm.add_constant(t.D_base.values), M=sm.robust.norms.HuberT()).fit(); print(f"  Huber M-estimator: slope {hub.params[1]:+.3f} se {hub.bse[1]:.3f}")
print(f"  quadratic term (identity predicts convexity): ", end="")
r = sm.OLS(t.ae_nb.values, sm.add_constant(np.column_stack([t.D_base, t.D_base**2]))).fit(cov_type="HC1"); print(f"D {r.params[1]:+.3f} (p={r.pvalues[1]:.3f}) D^2 {r.params[2]:+.3f} (p={r.pvalues[2]:.3f})")
rows = []
for s_ in range(2012, 2026):
    d = m[m.season == s_]; b, se, p = slope(d.ae_nb, d.D_base); bm = slope(d.ae_mkt, d.D_base)[0]
    rows.append(dict(season=s_, slope_model=b, slope_mkt=bm, diff=b - bm))
rr = pd.DataFrame(rows).set_index("season"); print("  per-season slopes (model vs market, same D):"); print(rr.round(3).T.to_string())
print(f"  model slope > market slope in {(rr['diff']>0).sum()}/{len(rr)} seasons (binom p={stats.binomtest(int((rr['diff']>0).sum()), len(rr)).pvalue:.3f}); mean diff {rr['diff'].mean():+.3f}")
