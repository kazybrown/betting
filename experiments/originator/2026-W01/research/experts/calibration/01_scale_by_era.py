"""Theory 1: is 25 Elo per point the right nfelo -> margin scale? Linearity for big favorites?
Lines in ORIGINATOR convention (negative = home favored). proj = -nfelo_lin = projected HOME margin."""
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from common import load, boot_ci, paired_mae_test
pd.set_option("display.width", 200)
m = load()
m["proj"] = -m.nfelo_lin          # projected home margin at 25 Elo/pt
m["proj_mkt"] = -m.mkt
m["proj_own"] = -m.nfelo_own
ERAS = {"2009-13": (2009, 2013), "2014-18": (2014, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025),
        "train<=2021": (2009, 2021), "all": (2009, 2025)}

def ols_slope(df, x, y="margin", const=False):
    X = df[[x]].values.astype(float)
    if const: X = sm.add_constant(X)
    r = sm.OLS(df[y].values, X).fit(cov_type="HC1")
    b = r.params[-1]; se = r.bse[-1]
    return b, se, (r.params[0] if const else np.nan), r.nobs

def qr_slope(df, x, y="margin", q=0.5):
    X = sm.add_constant(df[[x]].values.astype(float))
    r = QuantReg(df[y].values, X).fit(q=q)
    return r.params[1], r.bse[1], r.params[0]

print("\n=== A. Linear scale by era: margin = b * proj  (b=1 <=> 25 Elo/pt; implied Elo/pt = 25/b) ===")
rows = []
for era, (a, b_) in ERAS.items():
    d = m[(m.season >= a) & (m.season <= b_)]
    b, se, _, n = ols_slope(d, "proj")
    bq, seq, cq = qr_slope(d, "proj")
    bc, sec, c, _ = ols_slope(d, "proj", const=True)
    bm, sem, _, _ = ols_slope(d, "proj_mkt")
    bo, seo, _, _ = ols_slope(d, "proj_own")
    rows.append(dict(era=era, n=int(n), b_ols=b, ci=f"[{b-1.96*se:.3f},{b+1.96*se:.3f}]", elo_per_pt=25/b,
                     b_median=bq, ci_med=f"[{bq-1.96*seq:.3f},{bq+1.96*seq:.3f}]", elo_per_pt_med=25/bq,
                     b_with_const=bc, const=c, b_market=bm, ci_mkt=f"[{bm-1.96*sem:.3f},{bm+1.96*sem:.3f}]", b_nfelo_own=bo))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. Component scale (REG season, decomposition-verified rows): margin = b1*elo_dif_pts + b2*hfa_pts + b3*qb_pts ===")
reg = m[(~m.post) & m.decomp_ok]
rows = []
for era, (a, b_) in ERAS.items():
    d = reg[(reg.season >= a) & (reg.season <= b_)]
    X = d[["elo_dif_pts", "hfa_pts", "qb_pts"]].values
    r = sm.OLS(d.margin.values, X).fit(cov_type="HC1")
    r2 = sm.OLS(d.margin.values, sm.add_constant(X)).fit(cov_type="HC1")
    rows.append(dict(era=era, n=int(r.nobs), b_elo=r.params[0], se_elo=r.bse[0], b_hfa=r.params[1], se_hfa=r.bse[1],
                     b_qb=r.params[2], se_qb=r.bse[2], const_alt=r2.params[0], b_elo_alt=r2.params[1], b_hfa_alt=r2.params[2], b_qb_alt=r2.params[3]))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== C. Calibration by size of projected favorite (all seasons; realized = favorite's margin) ===")
def bins_table(df, projcol, label):
    fav = np.sign(df[projcol]); fav[fav == 0] = 1
    p = (df[projcol] * fav).values; r = (df.margin * fav).values
    cuts = [0, 3, 7, 10, 14, 99]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        k = (p >= lo) & (p < hi)
        if k.sum() < 20: continue
        lo_ci, hi_ci = boot_ci(r[k])
        out.append(dict(model=label, bin=f"[{lo},{hi})", n=int(k.sum()), mean_proj=p[k].mean(), mean_real=r[k].mean(),
                        ci_real=f"[{lo_ci:.2f},{hi_ci:.2f}]", ratio=r[k].mean()/p[k].mean() if p[k].mean() > 0 else np.nan,
                        median_real=np.median(r[k]), fav_covers=(r[k] > p[k]).mean(), mae=np.abs(r[k]-p[k]).mean()))
    return pd.DataFrame(out)
print(pd.concat([bins_table(m, "proj", "nfelo 25/pt"), bins_table(m, "proj_mkt", "market"), bins_table(m, "proj_own", "nfelo own map")]).round(3).to_string(index=False))
print("\n-- same, test seasons 2022-25 only --")
t = m[m.test]
print(pd.concat([bins_table(t, "proj", "nfelo 25/pt"), bins_table(t, "proj_mkt", "market")]).round(3).to_string(index=False))

print("\n=== C2. Piecewise: margin = b*proj + b7*hinge7 + b10*hinge10 (hinge_k = sign(proj)*max(|proj|-k,0)) ===")
def hinges(df, col="proj"):
    s = np.sign(df[col]); a = df[col].abs()
    return np.column_stack([df[col].values, (s*np.clip(a-7, 0, None)).values, (s*np.clip(a-10, 0, None)).values])
for era, (a, b_) in {"train<=2021": (2009, 2021), "2022-25": (2022, 2025), "all": (2009, 2025)}.items():
    d = m[(m.season >= a) & (m.season <= b_)]
    r = sm.OLS(d.margin.values, hinges(d)).fit(cov_type="HC1")
    rq = QuantReg(d.margin.values, hinges(d)).fit(q=0.5)
    print(f"{era:12s} n={int(r.nobs)} OLS b={r.params[0]:.3f}({r.bse[0]:.3f}) b7={r.params[1]:.3f}({r.bse[1]:.3f}) b10={r.params[2]:.3f}({r.bse[2]:.3f}) p7={r.pvalues[1]:.3f} p10={r.pvalues[2]:.3f}"
          f" | median-reg b={rq.params[0]:.3f} b7={rq.params[1]:.3f}({rq.bse[1]:.3f}) b10={rq.params[2]:.3f}({rq.bse[2]:.3f})")

print("\n=== D. OUT-OF-SAMPLE 2022-2025: MAE of candidate scales (fit on <=2021 where fitted) ===")
tr, te = m[m.train], m[m.test]
b_ols, _, _, _ = ols_slope(tr, "proj")
b_med, _, _ = qr_slope(tr, "proj")
rp = sm.OLS(tr.margin.values, hinges(tr)).fit()
rpq = QuantReg(tr.margin.values, hinges(tr)).fit(q=0.5)
cands = {"nfelo 25 Elo/pt (baseline)": te.proj.values,
         f"OLS scale b={b_ols:.3f} ({25/b_ols:.1f} Elo/pt)": b_ols*te.proj.values,
         f"median scale b={b_med:.3f} ({25/b_med:.1f} Elo/pt)": b_med*te.proj.values,
         "piecewise OLS (train)": hinges(te) @ rp.params,
         "piecewise median (train)": hinges(te) @ rpq.params,
         "nfelo own mapping (~27/pt, rounded)": te.proj_own.values,
         "market close": te.proj_mkt.values}
for elo_pt in (26, 27, 28, 30):
    cands[f"fixed {elo_pt} Elo/pt"] = te.proj.values * 25/elo_pt
base = te.margin.values - te.proj.values
rows = []
for k, v in cands.items():
    e = te.margin.values - v
    dm, lo, hi, p, n = paired_mae_test(e, base)
    rows.append(dict(candidate=k, n=n, MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), dMAE_vs_base=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                     bias=e.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== D2. Rolling-origin (fit scale on all seasons < t, test season t, t=2013..2025) ===")
rows = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]
    b, _, _, _ = ols_slope(trn, "proj"); bq, _, _ = qr_slope(trn, "proj")
    rows.append(dict(season=t_, n=len(tst), b_ols=b, b_med=bq, mae_base=np.abs(tst.margin-tst.proj).mean(),
                     mae_ols=np.abs(tst.margin-b*tst.proj).mean(), mae_med=np.abs(tst.margin-bq*tst.proj).mean(),
                     mae_27=np.abs(tst.margin-tst.proj*25/27).mean(), mae_mkt=np.abs(tst.margin-tst.proj_mkt).mean()))
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
print("rolling totals: base %.3f | ols-scaled %.3f | median-scaled %.3f | fixed 27 %.3f | market %.3f  (n=%d)" % tuple(
    [np.average(r[c], weights=r.n) for c in ["mae_base", "mae_ols", "mae_med", "mae_27", "mae_mkt"]] + [r.n.sum()]))
# paired test on the pooled rolling predictions
pool = []
for t_ in range(2013, 2026):
    trn = m[m.season < t_]; tst = m[m.season == t_]; b, _, _, _ = ols_slope(trn, "proj")
    pool.append(pd.DataFrame({"e_base": tst.margin - tst.proj, "e_ols": tst.margin - b*tst.proj, "e_27": tst.margin - tst.proj*25/27}))
pool = pd.concat(pool)
for c in ["e_ols", "e_27"]:
    dm, lo, hi, p, n = paired_mae_test(pool[c], pool.e_base)
    print(f"rolling pooled {c} vs base: dMAE={dm:.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")
