"""THEORY 2: best simple market-free totals model from local data, vs the market total.
Part A: choose blend constants K_team (team PF/PA prior weight, in games) and K_lg (league
        prior weight) on a VALIDATION window inside the training years (fit <=2014,
        validate 2015-2021). No test data touched.
Part B: OOS 2022-2025 (fit <=2021) ablation over feature groups; MAE / bias / O/U vs market
        with bootstrap CIs; drop-one importance; rolling-origin per season.
Part C: does the model carry information the market lacks? (residual regression, optimal
        blend weight fit on train and applied to test).
Model: OLS of (total_pts - lg_blend) on features, REG games only.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.api as sm
from common import build, mae, paired_mae_ci, ou_rate

pd.set_option("display.width", 200)


def lg_blend(m, K):
    return (K * m.lg_prev + m.n_before * m.lg_ytd.fillna(m.lg_prev)) / (K + m.n_before)


def fit_predict(tr, te, cols, offset="lg_blend"):
    X = sm.add_constant(tr[cols].astype(float), has_constant="add")
    y = tr.total_pts - tr[offset]
    ok = X.notna().all(axis=1) & y.notna()
    f = sm.OLS(y[ok], X[ok]).fit()
    Xt = sm.add_constant(te[cols].astype(float), has_constant="add")
    p = te[offset] + f.predict(Xt)
    return f, p


# ---------------- Part A: K selection on validation window ----------------
print("== Part A: K selection (fit 1999-2014, validate 2015-2021, REG games; model total-lg_blend ~ pf_sum + pa_sum) ==")
rows = []
for Kt in (1, 2, 3, 4, 6, 10, 16, 32):
    m = build(K_team=Kt, verbose=False)
    m = m[(m.game_type == "REG") & m.lg_prev.notna()].copy()
    for Kl in (16, 32, 64, 128, 256, 100000):
        m["lgb"] = lg_blend(m, Kl)
        tr = m[m.season <= 2014]; va = m[(m.season >= 2015) & (m.season <= 2021)]
        f, p = fit_predict(tr, va, ["pf_sum", "pa_sum"], offset="lgb")
        rows.append(dict(K_team=Kt, K_lg=Kl, val_MAE=mae(p, va.total_pts), val_bias=float((p - va.total_pts).mean()), n=len(va)))
res = pd.DataFrame(rows)
print(res.pivot(index="K_team", columns="K_lg", values="val_MAE").round(3).to_string())
best = res.sort_values("val_MAE").iloc[0]
print(f"best: K_team={int(best.K_team)} K_lg={int(best.K_lg)} val MAE={best.val_MAE:.3f} (market val MAE={mae(va.mkt_total, va.total_pts):.3f}, n={int(best.n)})")
K_TEAM, K_LG = int(best.K_team), int(best.K_lg)
# sensitivity is flat? print range
print(f"range of val MAE across grid: {res.val_MAE.min():.3f} .. {res.val_MAE.max():.3f}")

# ---------------- Part B: OOS ablation ----------------
m = build(K_team=K_TEAM, K_lg=K_LG, verbose=False)
m = m[(m.game_type == "REG") & m.lg_prev.notna()].copy()
m["plays_sum"] = m.h_plays + m.a_plays
m["spp_avg"] = (m.h_sec_per_play + m.a_sec_per_play) / 2
m["nh_sum"] = m.h_no_huddle_rate + m.a_no_huddle_rate
m["pr_sum"] = m.h_pass_rate + m.a_pass_rate
m["epa_off_sum"] = m.h_off_epa + m.a_off_epa
m["epa_def_sum"] = m.h_def_epa + m.a_def_epa
m["wind15"] = (m.wind_f >= 15).astype(int); m["wind20"] = (m.wind_f >= 20).astype(int)
m["cold32"] = (m.temp_f < 32).astype(int); m["cold20"] = (m.temp_f < 20).astype(int)
m["late"] = (m.week >= 14).astype(int)
m["dome"] = m.is_dome.astype(int)
m["wk"] = m.week.astype(float)

GROUPS = {
    "E  elo_sum": ["elo_sum"],
    "S  pf_sum+pa_sum": ["pf_sum", "pa_sum"],
    "Q  qb_sum (538 QB adj)": ["qb_sum"],
    "V  env: dome,div,week,late,short/bye,neutral,grass": ["dome", "div", "wk", "late", "short_home", "short_away", "bye_home", "bye_away", "neutral", "grass"],
    "W  weather: wind_f,wind15,wind20,cold32,cold20": ["wind_f", "wind15", "wind20", "cold32", "cold20"],
}
d = m[m.elo_sum.notna()].copy()   # 2009+ so every group is available
tr, te = d[d.train], d[d.test]
print(f"\n== Part B: OOS 2022-2025 (fit 2009-2021 REG, n_train={len(tr)}, n_test={len(te)}; K_team={K_TEAM}, K_lg={K_LG}) ==")
mk_err = te.mkt_total - te.total_pts
print(f"  {'model':58s} {'MAE':>7s} {'bias':>6s}  dMAE vs market [95% CI]        O/U vs mkt")
print(f"  {'market close':58s} {mae(te.mkt_total, te.total_pts):7.3f} {mk_err.mean():+6.2f}")


def report(lab, p):
    dm, lo, hi, n = paired_mae_ci(p - te.total_pts, mk_err)
    w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
    print(f"  {lab:58s} {mae(p, te.total_pts):7.3f} {(p - te.total_pts).mean():+6.2f}  {dm:+.3f} [{lo:+.3f},{hi:+.3f}]   {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
    return mae(p, te.total_pts)


report("L  lg_blend only", te.lg_blend)
report("L  lg_prev only", te.lg_prev)
cum = []
results = {}
for k, cols in GROUPS.items():
    cum = cum + cols
    f, p = fit_predict(tr, te, cum)
    results[k] = report("L+" + k.split()[0] + "  cumulative: " + "+".join(g.split()[0] for g in list(GROUPS)[:list(GROUPS).index(k) + 1]), p)
allcols = sum(GROUPS.values(), [])
f_all, p_all = fit_predict(tr, te, allcols)
print("\n  full-model coefficients (fit 2009-2021):")
for k, v in f_all.params.items():
    print(f"     {k:12s} {v:+.3f}  (se {f_all.bse[k]:.3f}, p={f_all.pvalues[k]:.3f})")
print("\n  drop-one importance (OOS MAE when the group is removed from the full model; + = group helps):")
base = mae(p_all, te.total_pts)
for k, cols in GROUPS.items():
    rest = [c for c in allcols if c not in cols]
    f, p = fit_predict(tr, te, rest)
    print(f"     drop {k:50s} MAE={mae(p, te.total_pts):.3f}  delta={mae(p, te.total_pts) - base:+.3f}")
for c in allcols:
    rest = [x for x in allcols if x != c]
    f, p = fit_predict(tr, te, rest)
    print(f"     drop single {c:44s} MAE={mae(p, te.total_pts):.3f}  delta={mae(p, te.total_pts) - base:+.3f}")

# lean model: L + E + S + Q + dome + wind
LEAN = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "cold32", "div"]
f_lean, p_lean = fit_predict(tr, te, LEAN)
print("\n  LEAN model (L + elo_sum, pf_sum, pa_sum, qb_sum, dome, wind_f, cold32, div):")
report("LEAN", p_lean)
for k, v in f_lean.params.items():
    print(f"     {k:12s} {v:+.3f}  (se {f_lean.bse[k]:.3f}, p={f_lean.pvalues[k]:.3f})")

# pace: only 2009-2019 train, 2023-2025 test
PACE = ["plays_sum", "spp_avg", "nh_sum", "pr_sum"]
dp = d[d.plays_sum.notna()]
trp, tep = dp[dp.train], dp[dp.test]
print(f"\n  PACE subset (train 2009-2019 n={len(trp)}, test 2023-2025 n={len(tep)}):")
mk_err_p = tep.mkt_total - tep.total_pts
for lab, cols in [("LEAN (pace subset)", LEAN), ("LEAN + pace(plays,spp,no-huddle,pass-rate)", LEAN + PACE), ("LEAN + plays_sum only", LEAN + ["plays_sum"]),
                  ("LEAN + pace + EPA off/def", LEAN + PACE + ["epa_off_sum", "epa_def_sum"]), ("LEAN + EPA off/def only", LEAN + ["epa_off_sum", "epa_def_sum"])]:
    f, p = fit_predict(trp, tep, cols)
    dm, lo, hi, n = paired_mae_ci(p - tep.total_pts, mk_err_p)
    w, l, pu = ou_rate(p, tep.mkt_total, tep.total_pts)
    print(f"  {lab:58s} MAE={mae(p, tep.total_pts):.3f} bias={(p - tep.total_pts).mean():+.2f}  dMAE vs mkt={dm:+.3f} [{lo:+.3f},{hi:+.3f}]  O/U {w}-{l}-{pu}  | market MAE={mae(tep.mkt_total, tep.total_pts):.3f}")
    if all(k in cols for k in PACE) and "EPA" not in lab:
        for k in PACE:
            print(f"       {k:10s} {f.params[k]:+.3f} (p={f.pvalues[k]:.3f})")

# rolling origin for LEAN
print("\n  rolling-origin LEAN (fit on REG seasons < Y):")
for Y in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025):
    a = d[d.season < Y]; b = d[d.season == Y]
    f, p = fit_predict(a, b, LEAN)
    print(f"     {Y}: LEAN MAE={mae(p, b.total_pts):.3f} bias={(p-b.total_pts).mean():+.2f} | market MAE={mae(b.mkt_total, b.total_pts):.3f} | spec 46+.35elo MAE={mae(46+.35*b.elo_sum, b.total_pts):.3f} | n={len(b)}")

# ---------------- Part C: information beyond the market ----------------
print("\n== Part C: does the LEAN model carry information the market lacks? ==")
for lab, (a, b, pa, pb) in {"train (in-sample)": (tr, tr, fit_predict(tr, tr, LEAN)[1], None), "test 2022-2025": (tr, te, p_lean, None)}.items():
    x = (pa - b.mkt_total); y = b.total_pts - b.mkt_total
    X = sm.add_constant(x.values.astype(float))
    r = sm.OLS(y.values.astype(float), X).fit(cov_type="HC1")
    print(f"  {lab:18s} (total - mkt) ~ (model - mkt): slope={r.params[1]:+.3f} (se {r.bse[1]:.3f}, p={r.pvalues[1]:.3f})  n={len(x)}  "
          f"corr={np.corrcoef(x, y)[0,1]:+.3f}  mean|model-mkt|={x.abs().mean():.2f}")
# optimal blend weight w on train (rolling: fit weight on seasons <= 2021 using LEAN fitted on <=2014? keep simple: in-sample w on train, apply to test)
p_tr = fit_predict(tr, tr, LEAN)[1]
ws = np.arange(0, 1.01, 0.05)
tr_m = [mae(w * p_tr + (1 - w) * tr.mkt_total, tr.total_pts) for w in ws]
w_best = ws[int(np.argmin(tr_m))]
te_m = [mae(w * p_lean + (1 - w) * te.mkt_total, te.total_pts) for w in ws]
print(f"  blend w*model + (1-w)*market: train-optimal w={w_best:.2f} -> test MAE={te_m[int(np.argmin(tr_m))]:.3f} vs market {te_m[0]:.3f}; "
      f"test-optimal w={ws[int(np.argmin(te_m))]:.2f} (MAE {min(te_m):.3f})")
print("  test MAE by w:", {round(w, 2): round(v, 3) for w, v in zip(ws, te_m) if round(w * 20) % 4 == 0})
# large disagreements
for thr in (2, 3, 4):
    sel = (p_lean - te.mkt_total).abs() >= thr
    w, l, pu = ou_rate(p_lean[sel], te.mkt_total[sel], te.total_pts[sel])
    print(f"  |model - market| >= {thr}: n={int(sel.sum())}  O/U taking model side {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
