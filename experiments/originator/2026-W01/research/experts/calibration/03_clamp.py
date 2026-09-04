"""Theory 3: is the structural clamp (pull engine X to within 4.5 of nfelo) sensible?
Proxy A: X = FiveThirtyEight qbelo line (independent engine), Y = nfelo_lin (25 Elo/pt).
Proxy B: X = nfelo_lin, Y = market close (does a big engine-vs-market gap predict who is right?)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import load, boot_ci, paired_mae_test
pd.set_option("display.width", 220)
m = load()
tr, te = m[m.train], m[m.test]

def weight_in_bins(d, X, Y, cuts=(0, 2, 4.5, 99), label=""):
    """Within |X-Y| bins, optimal weight beta on X given Y: err_Y = -beta*(X-Y). Also MAE of each and who is closer."""
    out = []
    gap = (d[X] - d[Y])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        k = (gap.abs() >= lo) & (gap.abs() < hi)
        dd = d[k]
        if len(dd) < 15: continue
        x = -(dd[X] - dd[Y]).values.astype(float); y = (dd.margin + dd[Y]).values
        r = sm.OLS(y, x).fit(cov_type="HC1")
        eX, eY = (dd.margin + dd[X]).abs(), (dd.margin + dd[Y]).abs()
        eMid = (dd.margin + 0.5*dd[X] + 0.5*dd[Y]).abs()
        closerX = (eX < eY).mean()
        out.append(dict(pair=label, gap_bin=f"[{lo},{hi})", n=len(dd), mean_gap=gap.abs()[k].mean(), beta_X=r.params[0], se=r.bse[0],
                        ci=f"[{r.params[0]-1.96*r.bse[0]:.2f},{r.params[0]+1.96*r.bse[0]:.2f}]", MAE_X=eX.mean(), MAE_Y=eY.mean(), MAE_mid=eMid.mean(),
                        share_X_closer=closerX, p_closer_vs_50=stats.binomtest(int((eX < eY).sum()), int(len(dd))).pvalue))
    return pd.DataFrame(out)

print("\n=== Proxy A: qbelo (X) vs nfelo_lin (Y): who is right when they disagree? ===")
for per, d in {"all 2009-25": m, "train 2009-21": tr, "test 2022-25": te}.items():
    print(f"-- {per} --"); print(weight_in_bins(d, "qbelo", "nfelo_lin", label="qbelo|nfelo").round(3).to_string(index=False))
print("\n-- finer bins, all seasons --")
print(weight_in_bins(m, "qbelo", "nfelo_lin", cuts=(0, 1, 2, 3, 4.5, 6, 99), label="qbelo|nfelo").round(3).to_string(index=False))

print("\n=== A2. Clamp backtest: blend = 0.541*nfelo_lin + 0.459*X (ORIGINATOR nfelo:PFF weights renormalised), X=qbelo ===")
def clamp(x, y, k):
    return np.clip(x, y - k, y + k)
def soft(x, y, k, keep=0.5):
    ex = x - y; s = np.sign(ex) * (np.minimum(np.abs(ex), k) + keep*np.clip(np.abs(ex)-k, 0, None)); return y + s
WN, WX = 0.46/(0.46+0.39), 0.39/(0.46+0.39)
def clamp_table(d, X="qbelo", Y="nfelo_lin"):
    base = d.margin + WN*d[Y] + WX*d[X]
    rows = [dict(variant="Y (nfelo_lin) alone", MAE=(d.margin+d[Y]).abs().mean()), dict(variant="X (qbelo) alone", MAE=(d.margin+d[X]).abs().mean()),
            dict(variant="blend, no clamp", MAE=base.abs().mean(), dMAE=0.0)]
    for k in (2, 3, 4.5, 6, 8):
        e = d.margin + WN*d[Y] + WX*clamp(d[X], d[Y], k)
        dm, lo, hi, p, n = paired_mae_test(e.values, base.values)
        rows.append(dict(variant=f"hard clamp k={k}", MAE=np.abs(e).mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p, n_clamped=int(((d[X]-d[Y]).abs() > k).sum())))
    for k in (3, 4.5):
        e = d.margin + WN*d[Y] + WX*soft(d[X], d[Y], k)
        dm, lo, hi, p, n = paired_mae_test(e.values, base.values)
        rows.append(dict(variant=f"soft clamp k={k} (keep 50% of excess)", MAE=np.abs(e).mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p, n_clamped=int(((d[X]-d[Y]).abs() > k).sum())))
    rows.append(dict(variant="market close", MAE=d.err_mkt.abs().mean()))
    return pd.DataFrame(rows)
for per, d in {"test 2022-25 (no parameters fitted -> OOS)": te, "all 2009-25": m}.items():
    print(f"-- {per} --"); print(clamp_table(d).round(3).to_string(index=False))

print("\n=== A3. Does the clamp help when measured against the MARKET (does the clamped blend land closer to the close)? test 2022-25 ===")
for k in (None, 3, 4.5, 6):
    x = te.qbelo if k is None else clamp(te.qbelo, te.nfelo_lin, k)
    bl = WN*te.nfelo_lin + WX*x
    print(f"  k={k}: mean |blend - market| = {np.abs(bl-te.mkt).mean():.3f}")

print("\n=== Proxy B: nfelo_lin (X) vs market (Y): does a big gap predict which side is right? ===")
for per, d in {"all 2009-25": m, "test 2022-25": te}.items():
    print(f"-- {per} --"); print(weight_in_bins(d, "nfelo_lin", "mkt", cuts=(0, 1, 2, 3, 4.5, 99), label="nfelo|mkt").round(3).to_string(index=False))
print("-- qbelo vs market, all --"); print(weight_in_bins(m, "qbelo", "mkt", cuts=(0, 1, 2, 3, 4.5, 99), label="qbelo|mkt").round(3).to_string(index=False))

print("\n=== B2. Interaction test: is the optimal engine weight a function of |gap|?  err_Y = -b0*(X-Y) - b1*(X-Y)*|X-Y| ===")
for (X, Y, d, per) in [("qbelo", "nfelo_lin", m, "all"), ("qbelo", "nfelo_lin", tr, "train"), ("qbelo", "nfelo_lin", te, "test"), ("nfelo_lin", "mkt", m, "all"), ("nfelo_lin", "mkt", te, "test")]:
    g = (d[X] - d[Y]).values.astype(float)
    Xm = np.column_stack([-g, -g*np.abs(g)])
    r = sm.OLS((d.margin + d[Y]).values, Xm).fit(cov_type="HC1")
    print(f"  {X:9s}|{Y:9s} {per:5s} n={int(r.nobs)}  b0={r.params[0]:.3f}±{1.96*r.bse[0]:.3f}  b1(gap*|gap|)={r.params[1]:.4f}±{1.96*r.bse[1]:.4f} p={r.pvalues[1]:.3f}")

print("\n=== A4. Role swap: clamp the STRONGER engine (nfelo_lin) to the weaker one (qbelo) +/- k. If the clamp only helps when the clamped engine is the weaker one, its value depends on PFF-vs-nfelo quality, which we cannot measure. ===")
def clamp_table_swap(d, X="nfelo_lin", Y="qbelo"):
    base = d.margin + WX*d[Y] + WN*d[X]   # same weights: nfelo 0.541, qbelo 0.459
    rows = [dict(variant="blend, no clamp", MAE=base.abs().mean(), dMAE=0.0)]
    for k in (2, 3, 4.5, 6):
        e = d.margin + WX*d[Y] + WN*clamp(d[X], d[Y], k)
        dm, lo, hi, p, n = paired_mae_test(e.values, base.values)
        rows.append(dict(variant=f"hard clamp nfelo->qbelo k={k}", MAE=np.abs(e).mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p, n_clamped=int(((d[X]-d[Y]).abs() > k).sum())))
    # symmetric clamp: pull both toward their midpoint so that |X-Y| <= k
    for k in (3, 4.5):
        mid = 0.5*(d[X]+d[Y]); half = np.minimum((d[X]-d[Y]).abs(), k)/2*np.sign(d[X]-d[Y])
        e = d.margin + WX*(mid-half) + WN*(mid+half)
        dm, lo, hi, p, n = paired_mae_test(e.values, base.values)
        rows.append(dict(variant=f"symmetric clamp to midpoint k={k}", MAE=np.abs(e).mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p, n_clamped=int(((d[X]-d[Y]).abs() > k).sum())))
    return pd.DataFrame(rows)
for per, d in {"test 2022-25": te, "all 2009-25": m}.items():
    print(f"-- {per} --"); print(clamp_table_swap(d).round(3).to_string(index=False))

print("\n=== A5. Rolling-origin choice of k (pick k on seasons < t by MAE from {none,2,3,4.5,6,8}) for the qbelo->nfelo clamp ===")
ks = [None, 2, 3, 4.5, 6, 8]
def blend_err(d, k):
    x = d.qbelo if k is None else clamp(d.qbelo, d.nfelo_lin, k)
    return d.margin + WN*d.nfelo_lin + WX*x
pool = []; picks = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    kbest = ks[int(np.argmin([blend_err(trn, k).abs().mean() for k in ks]))]
    picks.append((t_, kbest))
    pool.append(pd.DataFrame({"e_pick": blend_err(tst, kbest), "e_none": blend_err(tst, None), "e_45": blend_err(tst, 4.5)}))
pool = pd.concat(pool); print("k picked per season:", picks)
for a in ("e_pick", "e_45"):
    dm, lo, hi, p, n = paired_mae_test(pool[a], pool.e_none)
    print(f"rolling pooled {a} vs no-clamp: MAE {np.abs(pool[a]).mean():.3f} vs {np.abs(pool.e_none).mean():.3f} dMAE={dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")
