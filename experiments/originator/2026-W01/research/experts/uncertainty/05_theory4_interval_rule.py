"""05_theory4_interval_rule.py - THEORY 4: a per-game error-SD rule and tag thresholds.
  Target: the error of the number the model would publish. Benchmark = market close; we model
  |e_mkt| (spread) and |e_tot| (total) and also |e_nb| (model number) as a function of:
  D_base (model-market disagreement), abs_mkt (favorite size), mkt_total, dome, early (week<=4),
  late (week>=15), playoff, D_mmove (market move). Two estimators fitted on <=2021:
    A) OLS on |err|  -> sigma = sqrt(pi/2)*E|err|   (normal assumption)
    B) Gamma GLM (log link) on err^2 -> sigma = sqrt(E[err^2])
  OOS 2022-25: Spearman(sigma_hat, |err|), decile calibration (pred sigma vs realized RMSE),
  Gaussian log-score vs constant-sigma, PIT/coverage of 50/80/90% intervals, and the realized
  RMSE of HIGH/MED/LOW tags defined by sigma_hat terciles. Includes a season-level drift check
  (does a trailing-3-season base RMSE beat the fit-era constant?).
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/05_theory4_interval_rule.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

m = build()
m["early"] = (m.wk <= 4).astype(int); m["late"] = ((m.wk >= 15) & (m.playoff == 0)).astype(int)
m["fav_big"] = (m.abs_mkt >= 7).astype(int)
m["wind_hi"] = (m.wind.fillna(0) >= 15).astype(int)
fit, test = m[m.era == "fit"].copy(), m[m.era == "test"].copy()
pd.set_option("display.width", 220)
C = np.sqrt(np.pi / 2)


def evaluate(err_test, sig_hat, sig_const, label):
    e = np.asarray(err_test, float); s = np.asarray(sig_hat, float); s0 = float(sig_const)
    ae = np.abs(e)
    rho, p = stats.spearmanr(s, ae)
    ll_var = np.mean(stats.norm.logpdf(e, 0, s)); ll_con = np.mean(stats.norm.logpdf(e, 0, s0))
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred_sigma": s, "ae": ae, "e2": e**2, "dec": dec}).groupby("dec").agg(n=("ae", "size"), pred_sigma=("pred_sigma", "mean"), realized_rmse=("e2", lambda x: np.sqrt(x.mean())), mean_abs=("ae", "mean"))
    cov = {c: float(np.mean(np.abs(e) <= stats.norm.ppf(0.5 + c / 2) * s)) for c in [0.5, 0.8, 0.9]}
    cov0 = {c: float(np.mean(np.abs(e) <= stats.norm.ppf(0.5 + c / 2) * s0)) for c in [0.5, 0.8, 0.9]}
    print(f"\n--- {label} --- n={len(e)}")
    print(f"  Spearman(sigma_hat, |err|) = {rho:.3f} (p={p:.4f}) | pred sigma range {s.min():.2f}-{s.max():.2f} (sd {s.std():.2f}) | const sigma {s0:.2f}")
    print(f"  Gaussian mean log-score: variable {ll_var:.5f} vs constant {ll_con:.5f} -> gain {ll_var-ll_con:+.5f} per game")
    print(f"  interval coverage variable-sigma: {cov} | constant: {cov0} (targets .5/.8/.9)")
    print("  decile calibration:"); print(cal.round(3).to_string())
    # tags by tercile of sigma_hat
    edges = np.quantile(s, [1/3, 2/3])
    tag = np.where(s <= edges[0], "HIGH", np.where(s <= edges[1], "MED", "LOW"))
    print(f"  tags by sigma_hat terciles (cuts {edges.round(2)}):")
    for k in ["HIGH", "MED", "LOW"]:
        ek = e[tag == k]; print(f"    {k}: n={len(ek)} pred_sigma={s[tag==k].mean():.2f} realized RMSE={np.sqrt((ek**2).mean()):.2f} mean|err|={np.abs(ek).mean():.2f}")
    ek1, ek3 = e[tag == "HIGH"], e[tag == "LOW"]
    F = (ek3**2).mean() / (ek1**2).mean(); pF = 1 - stats.f.cdf(F, len(ek3) - 1, len(ek1) - 1)
    print(f"    variance ratio LOW/HIGH = {F:.3f} (F-test one-sided p={pF:.4f})")
    return dict(label=label, n=len(e), rho=rho, p=p, ll_gain=ll_var - ll_con, cov50=cov[0.5], cov80=cov[0.8], cov90=cov[0.9], var_ratio=F, pF=pF)


def fit_models(target_e, feats, name):
    yf = fit[target_e].values; Xf = sm.add_constant(fit[feats].astype(float).values)
    Xt = sm.add_constant(test[feats].astype(float).values)
    A = sm.OLS(np.abs(yf), Xf).fit(cov_type="HC1")
    B = sm.GLM(yf**2, Xf, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    print(f"\n##### {name}: features {feats}")
    coef = pd.DataFrame({"OLS|err| coef": A.params, "p": A.pvalues, "Gamma err^2 coef": B.params, "p_g": B.pvalues}, index=["const"] + feats)
    print(coef.round(4).to_string())
    sigA = C * np.clip(A.predict(Xt), 1.0, None); sigB = np.sqrt(B.predict(Xt))
    s0 = np.sqrt((yf**2).mean())
    resA = evaluate(test[target_e].values, sigA, s0, f"{name} / A: OLS on |err|")
    resB = evaluate(test[target_e].values, sigB, s0, f"{name} / B: Gamma on err^2")
    return A, B, resA, resB


print("=" * 100); print("SPREAD, target = market error e_mkt (benchmark number)")
feats_full = ["D_base", "abs_mkt", "mkt_total", "dome", "early", "late", "playoff", "D_mmove"]
A1, B1, rA1, rB1 = fit_models("e_mkt", feats_full, "e_mkt full")
A2, B2, rA2, rB2 = fit_models("e_mkt", ["D_base", "abs_mkt", "mkt_total", "dome"], "e_mkt simple")
print("\n" + "=" * 100); print("SPREAD, target = nfelo-base error e_nb (the model's own number)")
A3, B3, rA3, rB3 = fit_models("e_nb", feats_full, "e_nb full")
A4, B4, rA4, rB4 = fit_models("e_nb", ["D_base", "abs_mkt", "mkt_total", "dome"], "e_nb simple")
print("\n" + "=" * 100); print("TOTAL, target = market total error e_tot")
feats_tot = ["D_tot", "mkt_total", "abs_mkt", "dome", "early", "late", "playoff", "wind_hi"]
A5, B5, rA5, rB5 = fit_models("e_tot", feats_tot, "e_tot full")
A6, B6, rA6, rB6 = fit_models("e_tot", ["D_tot", "mkt_total", "dome"], "e_tot simple")

print("\n" + "=" * 100); print("Season-level DRIFT: realized RMSE by season vs fit-era constant; does a trailing-3-season base help?")
by = m.groupby("season").agg(n=("e_mkt", "size"), rmse_spread=("e_mkt", lambda x: np.sqrt((x**2).mean())), rmse_total=("e_tot", lambda x: np.sqrt((x**2).mean())), rmse_nb=("e_nb", lambda x: np.sqrt((x**2).mean())))
print(by.round(3).to_string())
s0 = np.sqrt((fit.e_mkt**2).mean()); rows = []
for s in range(2022, 2026):
    trail = m[(m.season >= s - 3) & (m.season < s)]; st = np.sqrt((trail.e_mkt**2).mean())
    t = m[m.season == s]
    ll0 = np.mean(stats.norm.logpdf(t.e_mkt, 0, s0)); ll1 = np.mean(stats.norm.logpdf(t.e_mkt, 0, st))
    st_t = np.sqrt((trail.e_tot**2).mean()); s0_t = np.sqrt((fit.e_tot**2).mean())
    ll0t = np.mean(stats.norm.logpdf(t.e_tot, 0, s0_t)); ll1t = np.mean(stats.norm.logpdf(t.e_tot, 0, st_t))
    rows.append(dict(season=s, n=len(t), fit_const=s0, trailing3=st, realized=np.sqrt((t.e_mkt**2).mean()), ll_gain_spread=ll1 - ll0, trailing3_tot=st_t, realized_tot=np.sqrt((t.e_tot**2).mean()), ll_gain_total=ll1t - ll0t))
print(pd.DataFrame(rows).set_index("season").round(4).to_string())

print("\n" + "=" * 100); print("SIMPLE IMPLEMENTABLE RULE (fit <=2021) with trailing-3-season base, evaluated 2022-25")
# spread: sigma = base_s * (1 + a*(D_base - mean) + b*(abs_mkt - mean)) ; fit a,b via OLS on |err| normalized by fit-era mean
def rule_eval(target, feats, label, base_col):
    f = fit; mu = {c: f[c].mean() for c in feats}
    X = sm.add_constant(np.column_stack([f[c] - mu[c] for c in feats]))
    A = sm.OLS(np.abs(f[target].values), X).fit(cov_type="HC1")
    rel = A.params[1:] / A.params[0]                     # relative sensitivity per unit of feature
    print(f"\n{label}: sigma = BASE * (1 + " + " + ".join(f"{rel[i]:+.4f}*({c}-{mu[c]:.2f})" for i, c in enumerate(feats)) + ")")
    out = []
    for s in range(2022, 2026):
        trail = m[(m.season >= s - 3) & (m.season < s)]; base = np.sqrt((trail[base_col]**2).mean())
        t = m[m.season == s]
        mult = 1 + sum(rel[i] * (t[c] - mu[c]) for i, c in enumerate(feats))
        sig = base * np.clip(mult, 0.6, 1.6)
        out.append(pd.DataFrame({"season": s, "e": t[target].values, "sig": sig.values, "base": base}))
    o = pd.concat(out)
    res = evaluate(o.e.values, o.sig.values, np.sqrt((fit[target]**2).mean()), f"{label} rule, trailing base")
    # tags at fixed multiplier thresholds instead of terciles: report realized RMSE for mult bins
    o["mult"] = o.sig / o.base
    print("  realized RMSE by multiplier bin:")
    for lo, hi in [(0, 0.95), (0.95, 1.05), (1.05, 99)]:
        e = o.e[(o.mult >= lo) & (o.mult < hi)]
        print(f"    mult in [{lo},{hi}): n={len(e)} share={len(e)/len(o):.2f} realized RMSE={np.sqrt((e**2).mean()):.2f}")
    return rel, mu, res

rel_s, mu_s, res_s = rule_eval("e_mkt", ["D_base", "abs_mkt", "mkt_total"], "SPREAD (market err)", "e_mkt")
rel_n, mu_n, res_n = rule_eval("e_nb", ["D_base", "abs_mkt", "mkt_total"], "SPREAD (nfelo-base err)", "e_nb")
rel_t, mu_t, res_t = rule_eval("e_tot", ["D_tot", "mkt_total", "dome"], "TOTAL (market err)", "e_tot")
print("\nSUMMARY of OOS evaluations:")
print(pd.DataFrame([rA1, rB1, rA2, rB2, rA3, rB3, rA4, rB4, rA5, rB5, rA6, rB6, res_s, res_n, res_t]).round(4).to_string())
