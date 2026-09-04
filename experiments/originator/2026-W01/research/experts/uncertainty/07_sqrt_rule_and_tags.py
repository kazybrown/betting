"""07_sqrt_rule_and_tags.py - the mechanical rule and the tag table.
Because e_model = e_mkt + d (d = model - market) and corr(e_mkt, d) ~ 0, the per-game RMSE of the
model's own number is sqrt(sigma_mkt^2 + d^2). Rule: sigma_g = sqrt(BASE_s^2 + D_g^2) with BASE_s =
trailing-3-season RMSE of the market close error (spread) / market total error (total).
OOS 2022-25 evaluation vs (i) fit-era constant, (ii) trailing base without D, (iii) OLS-on-|err|
model from 05. Empirical multipliers k_p = quantile(|e|/sigma, p) estimated on <=2021 give the
interval half-widths; OOS coverage is reported. Then the D-band tag table.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/07_sqrt_rule_and_tags.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, ats

pd.set_option("display.width", 220)
m = build()
fit, test = m[m.era == "fit"].copy(), m[m.era == "test"].copy()


def trailing_base(col, season, k=3):
    t = m[(m.season >= season - k) & (m.season < season)]
    return float(np.sqrt((t[col] ** 2).mean()))


def evaluate(e, sig, label, ks=None):
    e, sig = np.asarray(e, float), np.asarray(sig, float); ae = np.abs(e)
    rho, p = stats.spearmanr(sig, ae)
    ll = np.mean(stats.norm.logpdf(e, 0, sig))
    dec = pd.qcut(sig, 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"s": sig, "e2": e ** 2}).groupby(dec).agg(n=("s", "size"), pred_sigma=("s", "mean"), realized_rmse=("e2", lambda x: np.sqrt(x.mean())))
    out = f"{label:44s} n={len(e)} Spearman={rho:+.3f} (p={p:.3f}) logscore={ll:.5f}"
    if ks is not None:
        cov = {pp: float(np.mean(ae <= k * sig)) for pp, k in ks.items()}
        out += " | coverage " + " ".join(f"{pp}:{c:.3f}" for pp, c in cov.items())
    print(out); print("   quintile calibration (pred sigma -> realized RMSE): " + "; ".join(f"{r.pred_sigma:.2f}->{r.realized_rmse:.2f}" for r in cal.itertuples()))
    return ll, rho


for kind, ecol, dcol, base_col, title in [("SPREAD (model number = nfelo_b)", "e_nb", "D_base", "e_mkt", "spread"), ("TOTAL (model number = T_elo)", "e_telo", "D_tot", "e_tot", "total")]:
    print("\n" + "=" * 100); print(kind)
    # empirical multipliers from fit era using sqrt rule with fit-era in-sample base (per season trailing where possible)
    fit["base"] = [trailing_base(base_col, s) if s >= 2012 else np.sqrt((fit.loc[fit.season == s, base_col] ** 2).mean()) for s in fit.season]
    fit["sig"] = np.sqrt(fit.base ** 2 + fit[dcol] ** 2)
    z = (fit[ecol].abs() / fit.sig)
    ks = {p: float(np.quantile(z, p)) for p in [0.5, 0.8, 0.9]}
    kg = {p: stats.norm.ppf(0.5 + p / 2) for p in [0.5, 0.8, 0.9]}
    print(f"empirical multipliers k_p (fit <=2021, |e|/sigma quantiles): {dict((p, round(k, 3)) for p, k in ks.items())} | Gaussian: {dict((p, round(k, 3)) for p, k in kg.items())}")
    test["base"] = [trailing_base(base_col, s) for s in test.season]
    print("trailing-3 base by test season:", test.groupby("season").base.first().round(3).to_dict())
    const = float(np.sqrt((fit[base_col] ** 2).mean()))
    sig_const = np.full(len(test), const)
    sig_base = test.base.values
    sig_sqrt = np.sqrt(test.base.values ** 2 + test[dcol].values ** 2)
    # OLS-on-|err| comparator (fit-era) with D and base features as in 05 simple
    feats = [dcol, "abs_mkt", "mkt_total", "dome"]
    A = sm.OLS(fit[ecol].abs().values, sm.add_constant(fit[feats].astype(float).values)).fit()
    sig_ols = np.sqrt(np.pi / 2) * np.clip(A.predict(sm.add_constant(test[feats].astype(float).values)), 1, None)
    # sqrt rule + ols-scaled variant: multiply trailing base by OLS relative multiplier? keep simple.
    print("\nOOS 2022-25 evaluation (error of the model's own number):")
    r0 = evaluate(test[ecol], sig_const, "(i) fit-era constant sigma", ks)
    r1 = evaluate(test[ecol], sig_base, "(ii) trailing-3 base, no D", ks)
    r2 = evaluate(test[ecol], sig_sqrt, "(iii) sqrt(base^2 + D^2)  [PROPOSED]", ks)
    r3 = evaluate(test[ecol], sig_ols, "(iv) OLS on |err| (05 simple)", kg)
    print(f"log-score gain per game: (ii)-(i) {r1[0]-r0[0]:+.5f} | (iii)-(i) {r2[0]-r0[0]:+.5f} | (iii)-(ii) {r2[0]-r1[0]:+.5f} | (iv)-(i) {r3[0]-r0[0]:+.5f}")
    # paired bootstrap on per-game log-score difference (iii) vs (ii)
    e = test[ecol].values
    li = stats.norm.logpdf(e, 0, sig_sqrt) - stats.norm.logpdf(e, 0, sig_base)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(li), (3000, len(li))); bs = li[idx].mean(axis=1)
    print(f"  (iii) vs (ii) per-game log-score diff mean={li.mean():+.5f} 95% CI [{np.quantile(bs,.025):+.5f}, {np.quantile(bs,.975):+.5f}]")
    # for the market number itself: sqrt rule reduces to base; report coverage of base-only on e_mkt / e_tot
    print("Benchmark (market number) with trailing base only:")
    evaluate(test[base_col], sig_base, f"market {title}: trailing-3 base", ks)

    # TAG TABLE by D band
    print(f"\nTAG TABLE by {dcol} band (test 2022-25). expected excess RMSE = sqrt(base^2+D^2)-base at band mean D")
    if title == "spread":
        bands = [(0, 1.5, "HIGH"), (1.5, 3.0, "MED"), (3.0, 99, "LOW")]
    else:
        bands = [(0, 2.5, "HIGH"), (2.5, 5.0, "MED"), (5.0, 99, "LOW")]
    rows = []
    for lo, hi, name in bands:
        t = test[(test[dcol] >= lo) & (test[dcol] < hi)]
        b = t.base.mean(); Dm = t[dcol].mean(); Drms = np.sqrt((t[dcol] ** 2).mean())
        row = dict(tag=name, band=f"[{lo},{hi})", n=len(t), share=len(t) / len(test), D_mean=Dm, rmse_market=np.sqrt((t[base_col] ** 2).mean()), rmse_model=np.sqrt((t[ecol] ** 2).mean()),
                   predicted_model_rmse=np.sqrt(b ** 2 + Drms ** 2), excess_rmse_pred=np.sqrt(b ** 2 + Drms ** 2) - b, excess_rmse_realized=np.sqrt((t[ecol] ** 2).mean()) - np.sqrt((t[base_col] ** 2).mean()))
        if title == "spread":
            W, L, P = ats(t.nfelo_b, t.mkt, t.margin); row.update(ats_model_vs_mkt=f"{W}-{L}-{P}", ats_rate=W / (W + L), p_binom=stats.binomtest(W, W + L).pvalue)
        else:
            side = np.sign(t.T_elo - t.mkt_total); res_ = np.sign(t.total_pts - t.mkt_total); ok = (side != 0) & (res_ != 0)
            W = int(((side == res_) & ok).sum()); L = int(ok.sum()) - W; row.update(ats_model_vs_mkt=f"{W}-{L}", ats_rate=W / max(W + L, 1), p_binom=stats.binomtest(W, W + L).pvalue if W + L else np.nan)
        rows.append(row)
    print(pd.DataFrame(rows).set_index("tag").round(3).to_string())
    print("Same bands, FIT era (in-sample reference):")
    fit["base"] = [trailing_base(base_col, s) if s >= 2012 else np.sqrt((fit.loc[fit.season == s, base_col] ** 2).mean()) for s in fit.season]
    for lo, hi, name in bands:
        t = fit[(fit[dcol] >= lo) & (fit[dcol] < hi)]
        print(f"  {name} {[lo,hi]}: n={len(t)} share={len(t)/len(fit):.2f} rmse_market={np.sqrt((t[base_col]**2).mean()):.2f} rmse_model={np.sqrt((t[ecol]**2).mean()):.2f} predicted={np.sqrt(t.base.mean()**2 + (t[dcol]**2).mean()):.2f}")

print("\n" + "=" * 100); print("Homoscedasticity of the OUTCOME (market error) - joint test over all candidate drivers, full sample 2009-25 (n=%d)" % len(m))
m["early"] = (m.wk <= 4).astype(int); m["late"] = ((m.wk >= 15) & (m.playoff == 0)).astype(int); m["wind_hi"] = (m.wind.fillna(0) >= 15).astype(int)
for ecol, feats in [("e_mkt", ["D_base", "abs_mkt", "mkt_total", "dome", "early", "late", "playoff", "D_mmove"]), ("e_tot", ["D_tot", "mkt_total", "abs_mkt", "dome", "early", "late", "playoff", "wind_hi"])]:
    y = m[ecol].abs().values; X = sm.add_constant(m[feats].astype(float).values)
    r = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"{ecol}: |err| ~ features: F-test all slopes = 0: p={r.f_pvalue:.4f}; R^2={r.rsquared:.4f}; " + ", ".join(f"{f}={r.params[i+1]:+.3f}(p={r.pvalues[i+1]:.3f})" for i, f in enumerate(feats)))
    # Breusch-Pagan style on squared error
    r2 = sm.OLS(m[ecol].values ** 2, X).fit(cov_type="HC1"); print(f"   err^2 ~ features: F p={r2.f_pvalue:.4f}, R^2={r2.rsquared:.4f}")
