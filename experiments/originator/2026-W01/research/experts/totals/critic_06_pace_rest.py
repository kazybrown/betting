"""CRITIC 06 - T2 sub-claims: 'pace (pbp) hurts OOS' and 'rest/bye/week add nothing'. Expert used one split
(train 2009-2019, test 2023-25). Here: rolling origin over every pbp-available test season
(2012-2019 fit on 2009..Y-1; 2024-2025 fit on 2009-2019 + 2023..Y-1), pooled paired CIs vs FULL."""
import numpy as np, pandas as pd, statsmodels.api as sm
from critic_common import build_fixed, mae, paired_mae_ci

m = build_fixed(K_team=1, K_lg=128)
d = m[(m.game_type == "REG") & m.lg_prev.notna() & m.elo_sum.notna()].copy()
d["plays_sum"] = d.h_plays + d.a_plays; d["spp_avg"] = (d.h_sec_per_play + d.a_sec_per_play) / 2
d["nh_sum"] = d.h_no_huddle_rate + d.a_no_huddle_rate; d["pr_sum"] = d.h_pass_rate + d.a_pass_rate
d["epa_off_sum"] = d.h_off_epa + d.a_off_epa; d["epa_def_sum"] = d.h_def_epa + d.a_def_epa
d["wk"] = d.week.astype(float); d["late"] = (d.week >= 14).astype(int)
FULL = ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div", "dome", "wind_c", "cold20"]
PACE = ["plays_sum", "spp_avg", "nh_sum", "pr_sum"]; EPA = ["epa_off_sum", "epa_def_sum"]
REST = ["wk", "late", "short_home", "short_away", "bye_home", "bye_away", "neutral", "grass"]


def fp(a, b, cols):
    X = sm.add_constant(a[cols].astype(float), has_constant="add"); ok = X.notna().all(axis=1)
    f = sm.OLS((a.total_pts - a.lg_blend)[ok], X[ok]).fit()
    return b.lg_blend + f.predict(sm.add_constant(b[cols].astype(float), has_constant="add"))


print("== pace / EPA, rolling origin on pbp seasons ==")
dp = d[d.plays_sum.notna()]
SETS = {"FULL": FULL, "FULL+plays": FULL + ["plays_sum"], "FULL+pace4": FULL + PACE, "FULL+EPA": FULL + EPA, "FULL+pace4+EPA": FULL + PACE + EPA}
P = {k: [] for k in SETS}; Y_ = []; rows = []
for Y in list(range(2012, 2020)) + [2024, 2025]:
    a = dp[dp.season < Y]; b = dp[dp.season == Y]; Y_.append(b.total_pts.values); row = dict(Y=Y, n=len(b), market=mae(b.mkt_total, b.total_pts))
    for k, cols in SETS.items():
        p = fp(a, b, cols); P[k].append(p.values); row[k] = mae(p, b.total_pts)
    rows.append(row)
R = pd.DataFrame(rows).set_index("Y"); print(R.round(3).to_string()); print("mean:"); print(R.drop(columns="n").mean().round(3).to_string())
yv = np.concatenate(Y_); P = {k: np.concatenate(v) for k, v in P.items()}
for k in SETS:
    if k == "FULL": continue
    dm, lo, hi, n = paired_mae_ci(P[k] - yv, P["FULL"] - yv); print(f"  {k:16s} minus FULL: {dm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")

print("\n== rest/bye/week/late/neutral/grass, rolling origin 2016-2025 ==")
P = {"FULL": [], "FULL+REST": []}; Y_ = []
for Y in range(2016, 2026):
    a = d[d.season < Y]; b = d[d.season == Y]; Y_.append(b.total_pts.values)
    for k, cols in [("FULL", FULL), ("FULL+REST", FULL + REST)]:
        P[k].append(fp(a, b, cols).values)
yv = np.concatenate(Y_); P = {k: np.concatenate(v) for k, v in P.items()}
dm, lo, hi, n = paired_mae_ci(P["FULL+REST"] - yv, P["FULL"] - yv)
print(f"  FULL+REST minus FULL: {dm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}  (MAE FULL={np.abs(P['FULL']-yv).mean():.3f} FULL+REST={np.abs(P['FULL+REST']-yv).mean():.3f})")
a = d[d.season <= 2021]; X = sm.add_constant(a[FULL + REST].astype(float)); f = sm.OLS(a.total_pts - a.lg_blend, X).fit(cov_type="HC1")
print("  in-sample 2009-2021 REST coefficients: " + "  ".join(f"{k}={f.params[k]:+.2f}(p={f.pvalues[k]:.2f})" for k in REST))
