"""CRITIC 03 (T3: structural clamp). Alternative specifications:
A. does inter-engine disagreement |qbelo - nfelo| predict the error magnitude of nfelo, of the blend and of the MARKET?
   (relevant to the confidence-tag claim: disagreement = hard game vs disagreement = weaker engine is wrong)
B. the expert's proposed rule (anchor = engine with lower trailing-3-season MAE): which engine would it pick each season?
C. clamp vs simply lowering the second engine's weight: blend MAE across qbelo weights with/without the k=4.5 clamp;
D. placebo engines: (i) X = nfelo + pure noise (sd matched to the qbelo gap): the clamp must 'help' by construction;
   (ii) X = the market close (a BETTER engine): the clamp must hurt. Shows the clamp's sign is set by engine quality only.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import load, paired_mae_test
pd.set_option("display.width", 220)
m = load(verbose=False); tr, te = m[m.train], m[m.test]
WN, WX = 0.46 / (0.46 + 0.39), 0.39 / (0.46 + 0.39)
def clamp(x, y, k): return np.clip(x, y - k, y + k)

print("=== A. |gap| = |qbelo - nfelo_lin| as a predictor of |error| (OLS slope per pt of gap, HC1; Spearman) ===")
m["gap"] = (m.qbelo - m.nfelo_lin).abs(); m["err_blend"] = m.margin + WN * m.nfelo_lin + WX * m.qbelo
for per, d in {"all": m, "test": m[m.test]}.items():
    for c in ("err_nfelo_lin", "err_blend", "err_mkt", "err_nfelo_close"):
        r = sm.OLS(d[c].abs().values, sm.add_constant(d.gap.values)).fit(cov_type="HC1"); rho = stats.spearmanr(d.gap, d[c].abs())
        print(f"  {per:5s} |{c:15s}| ~ gap: slope {r.params[1]:+.3f} ± {1.96*r.bse[1]:.3f} p={r.pvalues[1]:.3f} | spearman {rho.statistic:+.3f} p={rho.pvalue:.3f}")
d = m.copy(); d["gbin"] = pd.cut(d.gap, [0, 1, 2, 3, 4.5, 99], right=False)
print(d.groupby("gbin", observed=True).agg(n=("gap", "size"), mae_nfelo=("err_nfelo_lin", lambda v: v.abs().mean()), mae_qbelo=("err_qbelo", lambda v: v.abs().mean()),
                                            mae_blend=("err_blend", lambda v: v.abs().mean()), mae_mkt=("err_mkt", lambda v: v.abs().mean()), mean_abs_mkt_line=("mkt", lambda v: v.abs().mean())).round(3).to_string())

print("\n=== B. Trailing-3-season MAE rule: which engine is the anchor each season? ===")
for t_ in range(2012, 2026):
    w = m[(m.season >= t_ - 3) & (m.season < t_)]
    a, b = w.err_nfelo_lin.abs().mean(), w.err_qbelo.abs().mean()
    print(f"  {t_}: trailing MAE nfelo {a:.3f} qbelo {b:.3f} -> anchor = {'nfelo' if a <= b else 'qbelo'}")

print("\n=== C. Clamp vs weight: blend = (1-wq)*nfelo + wq*qbelo, with and without qbelo clamped to nfelo±4.5 (paired vs nfelo alone) ===")
rows = []
for per, d in {"test 2022-25": te, "all 2009-25": m}.items():
    for wq in (0.459, 0.3, 0.2, 0.1, 0.0):
        for cl in (False, True):
            x = clamp(d.qbelo, d.nfelo_lin, 4.5) if cl else d.qbelo
            e = d.margin + (1 - wq) * d.nfelo_lin + wq * x
            dm, lo, hi, p, n = paired_mae_test(e.values, d.err_nfelo_lin.values)
            rows.append(dict(period=per, w_qbelo=wq, clamp=cl, MAE=np.abs(e).mean(), dMAE_vs_nfelo_alone=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== D. Placebo engines (test 2022-25 and all), blend 0.541 nfelo / 0.459 X, clamp X to nfelo±4.5 ===")
sd_gap = (m.qbelo - m.nfelo_lin).std(); print(f"  sd of (qbelo - nfelo_lin) = {sd_gap:.2f} pts; used as the noise sd")
rng = np.random.default_rng(1)
for per, d in {"test": te, "all": m}.items():
    res = []
    for s in range(200):
        xn = d.nfelo_lin.values + rng.normal(0, sd_gap, len(d))
        base = d.margin.values + WN * d.nfelo_lin.values + WX * xn
        cl = d.margin.values + WN * d.nfelo_lin.values + WX * clamp(xn, d.nfelo_lin.values, 4.5)
        res.append((np.abs(base).mean(), np.abs(cl).mean(), (np.abs(cl) - np.abs(base)).mean()))
    res = np.array(res)
    print(f"  {per:4s} X = nfelo + noise: blend MAE {res[:,0].mean():.3f} -> clamped {res[:,1].mean():.3f}; dMAE {res[:,2].mean():+.3f} (sd over 200 draws {res[:,2].std():.3f}); nfelo alone {d.err_nfelo_lin.abs().mean():.3f}")
    base = d.margin + WN * d.nfelo_lin + WX * d.mkt
    cl = d.margin + WN * d.nfelo_lin + WX * clamp(d.mkt, d.nfelo_lin, 4.5)
    dm, lo, hi, p, n = paired_mae_test(cl.values, base.values)
    print(f"  {per:4s} X = MARKET close: blend MAE {np.abs(base).mean():.3f} -> clamped {np.abs(cl).mean():.3f}; dMAE {dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n_clamped={int(((d.mkt-d.nfelo_lin).abs()>4.5).sum())}")
