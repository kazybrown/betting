"""04_theory3_who_is_right.py - THEORY 3: does disagreement predict WHO is right?
  Blend: line = mkt + w*(nfelo - mkt). Regress e_mkt (= margin + mkt) on x = -(nfelo - mkt) so the
  coefficient is w: w=0 -> market is right, w=1 -> model is right. Fit w on <=2021 overall, by
  D_base tercile, and with interaction w(D) = w0 + w1*D. Evaluate 2022-25: MAE/RMSE of the blend
  vs market, by tercile, with paired bootstrap CIs; ATS of model vs market by tercile.
  Done for the unregressed nfelo_b and the regressed nfelo_c.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/04_theory3_who_is_right.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, tercile_labels, ats, mae

m = build()
fit, test = m[m.era == "fit"], m[m.era == "test"]
pd.set_option("display.width", 200)
rng = np.random.default_rng(1)


def w_fit(d, model):
    x = -(d[model] - d.mkt)                      # model-minus-market in home-margin units
    r = sm.OLS(d.e_mkt.values, x.values).fit(cov_type="HC1")   # no intercept
    return r.params[0], r.bse[0], r.pvalues[0]


def paired_boot(a, b, B=3000):
    a, b = np.asarray(a), np.asarray(b); n = len(a)
    idx = rng.integers(0, n, size=(B, n)); diff = (a[idx] - b[idx]).mean(axis=1)
    return float(np.quantile(diff, .025)), float(np.quantile(diff, .975))


for model, name in [("nfelo_b", "nfelo UNREGRESSED (base)"), ("nfelo_c", "nfelo REGRESSED (close)")]:
    D = "D_base" if model == "nfelo_b" else "D_reg"
    print("\n" + "=" * 100); print(f"MODEL = {name}; disagreement = {D}")
    w, se, p = w_fit(fit, model)
    print(f"overall blend weight on model, fit era (n={len(fit)}): w={w:.3f} se={se:.3f} p={p:.4f}")
    wt, set_, pt = w_fit(test, model)
    print(f"overall blend weight on model, TEST era (n={len(test)}): w={wt:.3f} se={set_:.3f} p={pt:.4f}  [descriptive, not used for OOS eval]")
    # interaction
    x = -(fit[model] - fit.mkt); X = np.column_stack([x, x * fit[D]])
    r = sm.OLS(fit.e_mkt.values, X).fit(cov_type="HC1")
    print(f"interaction (fit): w(D) = {r.params[0]:.3f} + {r.params[1]:.3f}*D  (p_w1={r.pvalues[1]:.3f})")
    xt = -(test[model] - test.mkt); Xt = np.column_stack([xt, xt * test[D]])
    rt = sm.OLS(test.e_mkt.values, Xt).fit(cov_type="HC1")
    print(f"interaction (test, descriptive): w(D) = {rt.params[0]:.3f} + {rt.params[1]:.3f}*D  (p_w1={rt.pvalues[1]:.3f})")
    # tercile weights fit era, evaluate test era
    labf, edges = tercile_labels(fit[D].values)
    labt, _ = tercile_labels(test[D].values, edges=edges)
    print(f"tercile edges of {D} (fit era): {np.round(edges,2)}")
    rows = []
    for k in ["T1 low", "T2 mid", "T3 high"]:
        f = fit[labf == k]; t = test[labt == k]
        wk, sek, pk = w_fit(f, model)
        blend_line = t.mkt + wk * (t[model] - t.mkt)          # ORIGINATOR convention
        blend_glob = t.mkt + w * (t[model] - t.mkt)
        ae_m, ae_b, ae_g, ae_mod = t.ae_mkt.values, np.abs(t.margin + blend_line).values, np.abs(t.margin + blend_glob).values, np.abs(t.margin + t[model]).values
        lo, hi = paired_boot(ae_b, ae_m)
        W, L, P = ats(t[model], t.mkt, t.margin)
        pb = stats.binomtest(W, W + L, 0.5).pvalue if W + L > 0 else np.nan
        rows.append(dict(tercile=k, n_fit=len(f), w_fit=wk, se=sek, n_test=len(t), mae_mkt=ae_m.mean(), mae_model=ae_mod.mean(), mae_blend_tercile_w=ae_b.mean(), d_blend_minus_mkt=ae_b.mean() - ae_m.mean(), ci_lo=lo, ci_hi=hi, mae_blend_global_w=ae_g.mean(),
                         ats=f"{W}-{L}-{P}", ats_rate=W / (W + L), p_binom=pb))
    tab = pd.DataFrame(rows).set_index("tercile"); print(tab.round(4).to_string())
    # whole test set: blend with global w vs market
    bl = test.mkt + w * (test[model] - test.mkt); ae_b = np.abs(test.margin + bl).values
    lo, hi = paired_boot(ae_b, test.ae_mkt.values)
    rm_b, rm_m = np.sqrt((ae_b**2).mean()), np.sqrt((test.ae_mkt**2).mean())
    print(f"TEST whole: MAE market={test.ae_mkt.mean():.4f} blend(w={w:.2f})={ae_b.mean():.4f} diff={ae_b.mean()-test.ae_mkt.mean():+.4f} [95% CI {lo:+.4f},{hi:+.4f}] | RMSE market={rm_m:.4f} blend={rm_b:.4f}")
    # Disagreement-shrunk weight rule: w(D) from fit interaction, floored at 0
    wD = np.clip(r.params[0] + r.params[1] * test[D], 0, 1)
    bl2 = test.mkt + wD * (test[model] - test.mkt); ae_b2 = np.abs(test.margin + bl2).values
    lo2, hi2 = paired_boot(ae_b2, test.ae_mkt.values); lo3, hi3 = paired_boot(ae_b2, ae_b)
    print(f"TEST whole: blend with w(D) rule MAE={ae_b2.mean():.4f} vs market diff={ae_b2.mean()-test.ae_mkt.mean():+.4f} [95% CI {lo2:+.4f},{hi2:+.4f}] | vs constant-w blend diff={ae_b2.mean()-ae_b.mean():+.4f} [{lo3:+.4f},{hi3:+.4f}]")
    # Also: sign-of-disagreement test - when model and market disagree by > x, does the market side or model side cover?
    print("ATS of model side vs market by D bin (test era), with binomial p:")
    for lo_, hi_ in [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]:
        t = test[(test[D] >= lo_) & (test[D] < hi_)]
        if len(t) < 20:
            print(f"  D in [{lo_},{hi_}): n={len(t)} (too few)"); continue
        W, L, P = ats(t[model], t.mkt, t.margin); pb = stats.binomtest(W, W + L, 0.5).pvalue if W + L else np.nan
        print(f"  D in [{lo_},{hi_}): n={len(t)} ATS {W}-{L}-{P} rate={W/(W+L):.3f} p={pb:.3f} | MAE mkt={t.ae_mkt.mean():.3f} model={np.abs(t.margin+t[model]).mean():.3f}")
    print("Fit era same bins (in-sample, for reference):")
    for lo_, hi_ in [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]:
        t = fit[(fit[D] >= lo_) & (fit[D] < hi_)]
        if len(t) < 20: continue
        W, L, P = ats(t[model], t.mkt, t.margin); pb = stats.binomtest(W, W + L, 0.5).pvalue if W + L else np.nan
        print(f"  D in [{lo_},{hi_}): n={len(t)} ATS {W}-{L}-{P} rate={W/(W+L):.3f} p={pb:.3f} | MAE mkt={t.ae_mkt.mean():.3f} model={np.abs(t.margin+t[model]).mean():.3f}")

print("\n" + "=" * 100); print("Rolling-origin check of the constant blend weight w for nfelo_b (fit on all prior seasons >= 2009, test each season 2016-2025)")
rows = []
for s in range(2016, 2026):
    f = m[m.season < s]; t = m[m.season == s]
    w, se, p = w_fit(f, "nfelo_b")
    bl = t.mkt + w * (t.nfelo_b - t.mkt); ae_b = np.abs(t.margin + bl)
    rows.append(dict(season=s, n=len(t), w_fit=w, mae_mkt=t.ae_mkt.mean(), mae_blend=ae_b.mean(), diff=ae_b.mean() - t.ae_mkt.mean()))
tab = pd.DataFrame(rows).set_index("season"); print(tab.round(4).to_string())
print(f"mean diff (blend - market) 2016-25 = {tab['diff'].mean():+.4f}; seasons blend better: {(tab['diff']<0).sum()}/{len(tab)}")
