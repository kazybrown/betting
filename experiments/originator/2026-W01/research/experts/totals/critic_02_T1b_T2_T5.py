"""CRITIC 02 - T1b (PF/PA beyond Elo), T2/T5 (V3 model + ENV table value).
A. repaired join: refit V3 spec on 2009-2021 (n=3344 vs 3027) - coefficient stability, Huber, Wald pf=pa.
B. league-tracking confound: K_team=1 PF/PA blends react to the CURRENT season's league scoring
   level much faster than lg_blend(K=128). Is the 'S adds beyond Elo' gain team info or league info?
   Add an explicit fast league tracker and re-run the ablation.
C. rolling-origin ablation (fit REG seasons < Y, test Y) for Y=2016..2025 - 10 OOS years, not 4.
D. K_team rolling-origin (not the single 2015-21 validation window).
E. honest ENV value: refit-without-ENV baseline (constant re-estimated) vs ENV table; precip term
   (estimated on 2023-25 = TEST) removed.
F. rolling-origin refit of the full V3 spec vs the fixed rounded coefficients vs market.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from critic_common import build_fixed, mae, paired_mae_ci, ou_rate, rep, env_table, v3, C_V3

pd.set_option("display.width", 220)
FULL = ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div", "dome", "wind_c", "cold20"]


def fit(tr, cols, robust=False):
    X = sm.add_constant(tr[cols].astype(float), has_constant="add"); y = tr.total_pts - tr.lg_blend
    if robust:
        return sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return sm.OLS(y, X).fit(cov_type="HC1")


def pred(f, x, cols):
    return x.lg_blend + f.predict(sm.add_constant(x[cols].astype(float), has_constant="add"))


m = build_fixed(K_team=1, K_lg=128, verbose=True)
d = m[(m.game_type == "REG") & m.lg_prev.notna() & m.elo_sum.notna()].copy()
tr, te = d[d.train], d[d.test]
print(f"\n== A. repaired join: fit 2009-2021 n={len(tr)} (expert had 3027), test n={len(te)} ==")
f = fit(tr, FULL); fh = fit(tr, FULL, robust=True)
for k in f.params.index:
    print(f"   {k:8s} OLS={f.params[k]:+.3f} (se {f.bse[k]:.3f}, p={f.pvalues[k]:.3f})   Huber={fh.params[k]:+.3f} (se {fh.bse[k]:.3f})")
t = f.t_test("pf_dev - pa_dev = 0"); print(f"   Wald pf_dev - pa_dev = {float(np.squeeze(t.effect)):+.3f} p={float(np.squeeze(t.pvalue)):.2f}")
print("   OOS 2022-25 with refit coefficients (repaired join) vs expert's rounded V3 (no precip):")
rep("refit-on-repaired-join, fitted dome/wind/cold terms", pred(f, te, FULL), te)
rep("expert V3 rounded + ENV table (with precip)", v3(te), te)
rep("expert V3 rounded + ENV table (NO precip; precip was estimated on 2023-25 = test)", v3(te, env=env_table(te, precip=0.0)), te)
rep("market", te.mkt_total, te)

print("\n== B. league-tracking confound for the K_team=1 PF/PA term ==")
# fast league tracker: league ppg-sum blended with the same K=1 recipe using the average games played of the two teams
gp = (d.h_gp + d.a_gp) / 2.0
lg_fast = (1.0 * d.lg_prev + gp * d.lg_ytd.fillna(d.lg_prev)) / (1.0 + gp)
d["lg_gap"] = lg_fast - d.lg_blend                 # how far the fast league mean is from the slow prior (points per game)
d["pf_dev_t"] = d.pf_sum - lg_fast; d["pa_dev_t"] = d.pa_sum - lg_fast   # team deviations from the FAST league mean (pure team info)
tr, te = d[d.train], d[d.test]
print(f"   corr(lg_gap, pf_dev)={np.corrcoef(tr.lg_gap, tr.pf_dev)[0,1]:+.3f}  sd(lg_gap)={tr.lg_gap.std():.2f}")
for lab, cols in [("expert: elo + pf_dev + pa_dev", ["elo_sum", "pf_dev", "pa_dev"]),
                  ("elo + lg_gap only (league tracker, NO team scoring)", ["elo_sum", "lg_gap"]),
                  ("elo + pf_dev_t + pa_dev_t (team-only deviations)", ["elo_sum", "pf_dev_t", "pa_dev_t"]),
                  ("elo + pf_dev_t + pa_dev_t + lg_gap", ["elo_sum", "pf_dev_t", "pa_dev_t", "lg_gap"]),
                  ("elo only", ["elo_sum"])]:
    f = fit(tr, cols); p = pred(f, te, cols)
    co = "  ".join(f"{k}={f.params[k]:+.3f}({f.bse[k]:.3f})" for k in cols)
    print(f"   {lab:55s} OOS MAE={mae(p, te.total_pts):.3f} bias={(p-te.total_pts).mean():+.2f} | {co}")
f0 = fit(tr, ["elo_sum"]); p0 = pred(f0, te, ["elo_sum"])
for lab, cols in [("expert S term", ["elo_sum", "pf_dev", "pa_dev"]), ("team-only S term", ["elo_sum", "pf_dev_t", "pa_dev_t"]), ("league tracker only", ["elo_sum", "lg_gap"])]:
    f = fit(tr, cols); p = pred(f, te, cols); dm, lo, hi, n = paired_mae_ci(p - te.total_pts, p0 - te.total_pts)
    print(f"   paired vs elo-only: {lab:22s} dMAE={dm:+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n== C. rolling-origin ablation, fit REG < Y, test Y (repaired join; K_team=1, K_lg=128) ==")
SETS = {"L": [], "L+E": ["elo_sum"], "L+E+S": ["elo_sum", "pf_dev", "pa_dev"], "L+E+S+Q": ["elo_sum", "pf_dev", "pa_dev", "qb_sum"],
        "FULL": FULL, "FULL-S": [c for c in FULL if c not in ("pf_dev", "pa_dev")], "FULL-Q": [c for c in FULL if c != "qb_sum"],
        "FULL-ENV": [c for c in FULL if c not in ("dome", "wind_c", "cold20")], "FULL-E": [c for c in FULL if c != "elo_sum"]}
rows = []
for Y in range(2016, 2026):
    a = d[d.season < Y]; b = d[d.season == Y]; row = dict(Y=Y, n=len(b), market=mae(b.mkt_total, b.total_pts))
    for k, cols in SETS.items():
        if cols:
            f = fit(a, cols); p = pred(f, b, cols)
        else:
            p = b.lg_blend + (a.total_pts - a.lg_blend).mean()
        row[k] = mae(p, b.total_pts)
        if k == "FULL": row["FULL_bias"] = float((p - b.total_pts).mean())
    rows.append(row)
R = pd.DataFrame(rows).set_index("Y")
print(R.round(3).to_string())
print("   mean 2016-2025:"); print(R.drop(columns=["n"]).mean().round(3).to_string())
print("   mean 2022-2025:"); print(R.loc[2022:2025].drop(columns=["n"]).mean().round(3).to_string())
print("   per-year drop-one deltas (FULL-X minus FULL; + = feature helps):")
print((R[["FULL-S", "FULL-Q", "FULL-ENV", "FULL-E"]].sub(R["FULL"], axis=0)).round(3).to_string())
# paired CI for the drop-one over the pooled 2016-2025 rolling predictions
print("   pooled rolling 2016-2025 paired CIs (FULL vs FULL-X):")
P = {k: [] for k in SETS}; Yv = []
for Y in range(2016, 2026):
    a = d[d.season < Y]; b = d[d.season == Y]; Yv.append(b.total_pts.values)
    for k, cols in SETS.items():
        P[k].append((pred(fit(a, cols), b, cols) if cols else b.lg_blend + (a.total_pts - a.lg_blend).mean()).values)
yv = np.concatenate(Yv); P = {k: np.concatenate(v) for k, v in P.items()}
for k in ["FULL-S", "FULL-Q", "FULL-ENV", "FULL-E", "L+E"]:
    dm, lo, hi, n = paired_mae_ci(P[k] - yv, P["FULL"] - yv)
    print(f"     {k:9s} minus FULL: {dm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")

print("\n== D. K_team by rolling origin (fit < Y, test Y, 2016-2025), FULL spec ==")
for K in (1, 2, 3, 6, 10):
    mk = build_fixed(K_team=K, K_lg=128); dk = mk[(mk.game_type == "REG") & mk.lg_prev.notna() & mk.elo_sum.notna()]
    errs = []
    for Y in range(2016, 2026):
        a = dk[dk.season < Y]; b = dk[dk.season == Y]; errs.append((pred(fit(a, FULL), b, FULL) - b.total_pts).values)
    e = np.concatenate(errs); e22 = np.concatenate(errs[6:])
    print(f"   K_team={K:2d}: rolling MAE 2016-25={np.abs(e).mean():.3f}  2022-25={np.abs(e22).mean():.3f}")

print("\n== E. honest ENV value OOS 2022-25 (K_team=1): baseline must be REFIT without ENV terms, not the ENV-fitted constant with ENV zeroed ==")
d = m[(m.game_type == "REG") & m.lg_prev.notna() & m.elo_sum.notna()].copy(); tr, te = d[d.train], d[d.test]
NOENV = ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div"]
f_no = fit(tr, NOENV); p_no = pred(f_no, te, NOENV)
f_env = fit(tr, FULL); p_env = pred(f_env, te, FULL)
rep("refit WITHOUT env terms (baseline)", p_no, te, p_no)
rep("refit WITH fitted dome/wind_c/cold20", p_env, te, p_no)
rep("expert rounded V3 + ENV table (with precip)", v3(te), te, p_no)
rep("expert rounded V3 + ENV table (no precip)", v3(te, env=env_table(te, precip=0.0)), te, p_no)
rep("expert rounded V3, ENV zeroed (expert's 'no ENV' comparator)", v3(te, env=pd.Series(0.0, index=te.index)), te, p_no)
# ENV table value on the fitted-no-ENV baseline shifted to the same mean (pure shape test)
tab = env_table(te, precip=0.0); tab_c = tab - (env_table(tr, precip=0.0)).mean()
rep("refit-no-ENV + ENV table centered on its train mean (shape only)", p_no + tab_c, te, p_no)
# dome-only and wind-only pieces
rep("refit-no-ENV + dome piece only (+2.5 dome vs outdoor, centered)", p_no + (np.where(te.dome == 1, 2.5, 0.0) - np.where(tr.dome == 1, 2.5, 0.0).mean()), te, p_no)
wpiece = np.where(te.outdoor == 1, -0.2 * te.wind_c, 0.0); wpiece_tr = np.where(tr.outdoor == 1, -0.2 * tr.wind_c, 0.0)
rep("refit-no-ENV + wind piece only (-0.2/mph centered)", p_no + (wpiece - wpiece_tr.mean()), te, p_no)

print("\n== F. per-season: rolling refit of FULL spec vs fixed rounded V3 (no precip) vs market ==")
for Y in range(2019, 2026):
    a = d[d.season < Y]; b = d[d.season == Y]
    p_roll = pred(fit(a, FULL), b, FULL); p_fix = v3(b, env=env_table(b, precip=0.0))
    print(f"   {Y}: rolling-refit MAE={mae(p_roll, b.total_pts):.3f} bias={(p_roll-b.total_pts).mean():+.2f} | fixed V3 MAE={mae(p_fix, b.total_pts):.3f} bias={(p_fix-b.total_pts).mean():+.2f} | market {mae(b.mkt_total, b.total_pts):.3f} | n={len(b)}{'  <- coefficients in-sample for fixed V3' if Y <= 2021 else ''}")
p_fix = v3(te, env=env_table(te, precip=0.0))
x = p_fix - te.mkt_total; y = te.total_pts - te.mkt_total
r = sm.OLS(y.values, sm.add_constant(x.values)).fit(cov_type="HC1")
print(f"   info beyond market, V3(no precip) 2022-25: slope (total-mkt)~(model-mkt) = {r.params[1]:+.3f} (se {r.bse[1]:.3f}, p={r.pvalues[1]:.3f})")
