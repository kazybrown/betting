"""THEORY 5: the recommended totals formula + environment table, validated OOS.
Versions (all market-free):
  V0 spec     : 46.0 + 0.35*elo_sum
  V1 minimal  : LG + 0.28*elo_sum                     (LG = league prior, see below)
  V2 V1+env   : V1 + ENV(dome, wind, cold)             (weather/dome table)
  V3 full     : LG + 0.14*elo_sum + 0.26*(pf_sum - 4*lgppg) + 0.22*(pa_sum - 4*lgppg)
                + 0.80*qb_sum - 1.3*div + ENV
  V3g         : like V3 but one scoring term 0.24*(gt_sum - 4*lgppg) (simpler)
League prior LG options: prior-season REG mean (lg_prev), 2-season mean, K=128 blend with
season-to-date (lg_blend). Team scoring blend: K_team = 1 (prior-season mean counts as ONE
game vs the season-to-date mean).
ENV table (outdoor): wind 0-5 +0.5 | 6-9 -0.5 | 10-14 -1.5 | 15-19 -2.5 | 20-24 -3.5 | 25+ -5.0;
temp <20F -1.0; dome/closed +2.0 (= -0.5 - 0.2*(wind-8.4) binned; dome vs avg outdoor +2.5); precipitation -1.5 (2023-25 flag only; n too small to
validate, applied where the flag exists).
Coefficients are ROUNDED values from the 2009-2021 fit (02/03/04); this script re-fits the
exact values for reference, then evaluates the rounded formula OOS 2022-2025 and rolling.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.api as sm
from common import build, mae, paired_mae_ci, ou_rate

m = build(K_team=1, K_lg=128, verbose=False)
m = m[(m.game_type == "REG") & m.lg_prev.notna()].copy()
m["dome"] = m.is_dome.astype(int)
# league points-per-team-game prior (so scoring features are deviations from league average)
m["lgppg"] = m.lg_blend / 2.0
m["pf_dev"] = m.pf_sum - 2 * m.lgppg; m["pa_dev"] = m.pa_sum - 2 * m.lgppg
m["gt_dev"] = m.pf_dev + m.pa_dev
# 2-season league prior
lg2 = m.groupby("season").lg_prev.first()
m["lg_prev2"] = m.season.map(lambda s: np.nanmean([lg2.get(s, np.nan), lg2.get(s - 1, np.nan)]))
m["wind_c"] = np.where(m.outdoor == 1, m.wind_f - 8.4, 0.0)
m["cold20"] = ((m.outdoor == 1) & (m.temp_f < 20)).astype(int)


def env_table(df, precip=True):
    w = df.wind_f
    tab = np.select([w <= 5, w <= 9, w <= 14, w <= 19, w <= 24], [0.5, -0.5, -1.5, -2.5, -3.5], -5.0)
    tab = np.where(df.outdoor == 1, tab + np.where(df.temp_f < 20, -1.0, 0.0), 2.0)
    if precip:
        tab = tab + np.where((df.outdoor == 1) & (df.precip_strict.fillna(0) == 1), -1.5, 0.0)
    return pd.Series(tab, index=df.index)


m["env"] = env_table(m)
m["env_lin"] = np.where(m.outdoor == 1, -0.2 * m.wind_c - 0.5 - 1.0 * m.cold20, 2.0)

d = m[m.elo_sum.notna()].copy()
tr, te = d[d.train], d[d.test]
print(f"sample: REG games with Elo, fit 2009-2021 n={len(tr)}, test 2022-2025 n={len(te)}")

# ---------------- exact re-fit for reference ----------------
print("\n== exact coefficients, fit 2009-2021, y = total - lg_blend ==")
for lab, cols in [("V1 elo only", ["elo_sum"]), ("V3 pf/pa split", ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div", "dome", "wind_c", "cold20"]),
                  ("V3g single gt term", ["elo_sum", "gt_dev", "qb_sum", "div", "dome", "wind_c", "cold20"]),
                  ("V3 with env table as one regressor", ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div", "env"])]:
    X = sm.add_constant(tr[cols].astype(float)); f = sm.OLS(tr.total_pts - tr.lg_blend, X).fit(cov_type="HC1")
    print(f"  {lab:36s} " + "  ".join(f"{k}={v:+.3f}({f.bse[k]:.3f})" for k, v in f.params.items()))
    if lab.startswith("V3 pf/pa"):
        # is pf coefficient different from pa? (Wald)
        t = f.t_test("pf_dev - pa_dev = 0")
        eff, pv = float(np.squeeze(t.effect)), float(np.squeeze(t.pvalue))
        print(f"     pf_dev - pa_dev = {eff:+.3f} p={pv:.2f}  -> {'separate off/def NOT needed' if pv > 0.1 else 'keep separate'}")
        C = {k: round(float(v), 2) for k, v in f.params.items()}
    if lab.startswith("V3g"):
        C["gt_dev"] = round(float(f.params["gt_dev"]), 2)
print("  ROUNDED coefficients used below (from the V3 fits):", C)

# ---------------- rounded formulas ----------------
def formulas(x, LG):
    cE, cPF, cPA, cQ, cD, cG = C["elo_sum"], C["pf_dev"], C["pa_dev"], C["qb_sum"], C["div"], C["gt_dev"]
    out = {}
    out["V0 spec 46.0 + 0.35*elo"] = 46.0 + 0.35 * x.elo_sum
    out["V1 LG + 0.28*elo"] = LG + 0.28 * x.elo_sum
    out["V2 V1 + ENV table"] = LG + 0.28 * x.elo_sum + x.env
    out["V3 full (pf/pa split) + ENV"] = LG + cE * x.elo_sum + cPF * x.pf_dev + cPA * x.pa_dev + cQ * x.qb_sum + cD * x["div"] + x.env
    out["V3g full (single gt term) + ENV"] = LG + cE * x.elo_sum + cG * x.gt_dev + cQ * x.qb_sum + cD * x["div"] + x.env
    out["V3 full, ENV linear (-0.2/mph)"] = LG + cE * x.elo_sum + cPF * x.pf_dev + cPA * x.pa_dev + cQ * x.qb_sum + cD * x["div"] + x.env_lin
    out["V3 full, no ENV"] = LG + cE * x.elo_sum + cPF * x.pf_dev + cPA * x.pa_dev + cQ * x.qb_sum + cD * x["div"]
    out["V3 full, no QB"] = LG + cE * x.elo_sum + cPF * x.pf_dev + cPA * x.pa_dev + cD * x["div"] + x.env
    out["V3 full, no Elo"] = LG + cPF * x.pf_dev + cPA * x.pa_dev + cQ * x.qb_sum + cD * x["div"] + x.env
    return out


print("\n== OOS 2022-2025 (n=%d), league prior = lg_blend (K=128) ==" % len(te))
mk = te.mkt_total
print(f"  {'formula':40s} {'MAE':>7s} {'bias':>6s}   dMAE vs market [95% CI]    O/U vs market")
print(f"  {'market close':40s} {mae(mk, te.total_pts):7.3f} {(mk - te.total_pts).mean():+6.2f}")
for k, p in formulas(te, te.lg_blend).items():
    dm, lo, hi, n = paired_mae_ci(p - te.total_pts, mk - te.total_pts)
    w, l, pu = ou_rate(p, mk, te.total_pts)
    print(f"  {k:40s} {mae(p, te.total_pts):7.3f} {(p - te.total_pts).mean():+6.2f}   {dm:+.3f} [{lo:+.3f},{hi:+.3f}]   {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
# env table vs no env, paired
p_env = formulas(te, te.lg_blend)["V3 full (pf/pa split) + ENV"]; p_no = formulas(te, te.lg_blend)["V3 full, no ENV"]; p_lin = formulas(te, te.lg_blend)["V3 full, ENV linear (-0.2/mph)"]
dm, lo, hi, n = paired_mae_ci(p_env - te.total_pts, p_no - te.total_pts)
print(f"  ENV table vs no ENV (paired): dMAE={dm:+.3f} [{lo:+.3f},{hi:+.3f}]")
dm, lo, hi, n = paired_mae_ci(p_lin - te.total_pts, p_no - te.total_pts)
print(f"  ENV linear vs no ENV (paired): dMAE={dm:+.3f} [{lo:+.3f},{hi:+.3f}]")
sel = te.outdoor.eq(1) & (te.wind_f >= 15) & (te.wx_missing == 0)
print(f"  windy test games (>=15 mph observed, n={int(sel.sum())}): no ENV MAE={mae(p_no[sel], te.total_pts[sel]):.3f} table={mae(p_env[sel], te.total_pts[sel]):.3f} linear={mae(p_lin[sel], te.total_pts[sel]):.3f} market={mae(mk[sel], te.total_pts[sel]):.3f}; bias no ENV={(p_no[sel]-te.total_pts[sel]).mean():+.2f} table={(p_env[sel]-te.total_pts[sel]).mean():+.2f}")
sel = te.dome.eq(1)
print(f"  dome test games (n={int(sel.sum())}): no ENV MAE={mae(p_no[sel], te.total_pts[sel]):.3f} table={mae(p_env[sel], te.total_pts[sel]):.3f} market={mae(mk[sel], te.total_pts[sel]):.3f}; bias no ENV={(p_no[sel]-te.total_pts[sel]).mean():+.2f} table={(p_env[sel]-te.total_pts[sel]).mean():+.2f}")

# ---------------- league prior choice, rolling 2010-2025 ----------------
print("\n== league prior choice (V1 formula), per season MAE / bias (no fitting involved) ==")
print(f"  {'season':6s} {'46.0':>14s} {'lg_prev':>14s} {'lg_prev2':>14s} {'lg_blend128':>14s} {'market':>8s}  realized")
tot = {k: [] for k in ["46.0", "lg_prev", "lg_prev2", "lg_blend"]}
for Y in range(2010, 2026):
    x = d[d.season == Y]
    row = []
    for k, LG in [("46.0", pd.Series(46.0, index=x.index)), ("lg_prev", x.lg_prev), ("lg_prev2", x.lg_prev2), ("lg_blend", x.lg_blend)]:
        p = LG + 0.28 * x.elo_sum; e = p - x.total_pts
        row.append(f"{mae(p, x.total_pts):6.3f}/{e.mean():+5.2f}"); tot[k].append(mae(p, x.total_pts))
    print(f"  {Y:6d} " + " ".join(f"{r:>14s}" for r in row) + f" {mae(x.mkt_total, x.total_pts):8.3f}  {x.total_pts.mean():.1f}")
print("  mean  " + " ".join(f"{np.mean(v):>14.3f}" for v in tot.values()))

# ---------------- rolling-origin for the rounded formulas (no refit needed; coefficients fixed) ----------------
print("\n== rounded formulas by season (coefficients fixed from the 2009-2021 fit; seasons <=2021 are IN-SAMPLE for the coefficients) ==")
print(f"  {'season':6s} {'V0 spec':>8s} {'V1':>8s} {'V2':>8s} {'V3':>8s} {'V3g':>8s} {'market':>8s}   V3 O/U vs mkt")
for Y in range(2016, 2026):
    x = d[d.season == Y]; F = formulas(x, x.lg_blend)
    w, l, pu = ou_rate(F["V3 full (pf/pa split) + ENV"], x.mkt_total, x.total_pts)
    print(f"  {Y:6d} {mae(F['V0 spec 46.0 + 0.35*elo'], x.total_pts):8.3f} {mae(F['V1 LG + 0.28*elo'], x.total_pts):8.3f} {mae(F['V2 V1 + ENV table'], x.total_pts):8.3f} "
          f"{mae(F['V3 full (pf/pa split) + ENV'], x.total_pts):8.3f} {mae(F['V3g full (single gt term) + ENV'], x.total_pts):8.3f} {mae(x.mkt_total, x.total_pts):8.3f}   {w}-{l}-{pu} ({w/max(w+l,1):.3f}){'  <- OOS' if Y >= 2022 else ''}")

# ---------------- confidence: SD of model error by |model - market| and environment ----------------
print("\n== error scale (for the totals confidence tag): OOS 2022-2025 abs error of V3 ==")
e = (p_env - te.total_pts)
print(f"  overall MAE={e.abs().mean():.2f} RMSE={np.sqrt((e**2).mean()):.2f} SD={e.std():.2f}; market MAE={mae(mk, te.total_pts):.2f}")
for lab, s in [("dome", te.dome.eq(1)), ("outdoor wind<10", te.outdoor.eq(1) & (te.wind_f < 10)), ("outdoor wind 10-14", te.outdoor.eq(1) & (te.wind_f >= 10) & (te.wind_f < 15)), ("outdoor wind>=15", te.outdoor.eq(1) & (te.wind_f >= 15))]:
    print(f"  {lab:20s} n={int(s.sum()):4d} V3 MAE={e[s].abs().mean():.2f} RMSE={np.sqrt((e[s]**2).mean()):.2f} | market MAE={(mk - te.total_pts)[s].abs().mean():.2f}")

# ---------------- team-total identity sanity (spec: home = T/2 - S/2) ----------------
print("\n== team-total identity check with market lines (2009-2025 REG): home_pts vs T/2 - S/2 (S in ORIGINATOR convention, negative = home fav) ==")
x = d.dropna(subset=["mkt_spread"])
h_imp = x.mkt_total / 2 - x.mkt_spread / 2; a_imp = x.mkt_total / 2 + x.mkt_spread / 2
print(f"  home: MAE={mae(h_imp, x.home_score):.3f} bias={(h_imp - x.home_score).mean():+.2f} | away: MAE={mae(a_imp, x.away_score):.3f} bias={(a_imp - x.away_score).mean():+.2f}  n={len(x)}")
