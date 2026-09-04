"""CRITIC 01 - T1a (Elo slope 0.35 and constant 46.0).
Attack 1: the spec's 46.0 is BY CONSTRUCTION the prior-season realized mean (README: '46.0 = 2025
  realized mean'). Applying 46.0 retroactively to 2022-25 is a straw man; the fair counterfactual
  is lg_prev (prior-season mean) each year. Decompose the expert's dMAE into (i) level mechanism,
  (ii) slope 0.35->0.28, (iii) in-season blend.
Attack 2: repaired nfelo join (+317 training games) - does b change?
Attack 3: K_lg sensitivity, rolling with no fitting, paired vs lg_prev.
Attack 4: slope grid OOS; Huber slope.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from critic_common import build_fixed, build, mae, paired_mae_ci, rep

m0 = build(K_team=1, K_lg=128, verbose=False)            # expert's sample (broken join)
m1 = build_fixed(K_team=1, K_lg=128)                      # repaired join
for lab, m in [("expert join", m0), ("repaired join", m1)]:
    d = m[(m.game_type == "REG") & m.mkt_total.notna() & m.elo_sum.notna()]
    tr, te = d[d.train], d[d.test]
    f = smf.ols("total_pts ~ elo_sum + C(season)", data=tr).fit(cov_type="HC1")
    ci = f.conf_int().loc["elo_sum"]
    fh = sm.RLM(tr.total_pts - tr.lg_prev, sm.add_constant(tr.elo_sum), M=sm.robust.norms.HuberT()).fit()
    print(f"[{lab}] train n={len(tr)} test n={len(te)}: b={f.params['elo_sum']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | Huber b={fh.params['elo_sum']:.3f} (se {fh.bse['elo_sum']:.3f})")

d = m1[(m1.game_type == "REG") & m1.mkt_total.notna() & m1.elo_sum.notna()]
tr, te = d[d.train], d[d.test]
print(f"\n== OOS 2022-2025 n={len(te)}: decomposition of the expert's claimed -0.16 MAE (ref = lg_prev + 0.35*elo, i.e. what the spec's own mechanism produces each season) ==")
ref = te.lg_prev + 0.35 * te.elo_sum
rep("market close", te.mkt_total, te, ref)
rep("fixed 46.0 + 0.35*elo  (expert's 'spec' straw man)", 46.0 + 0.35 * te.elo_sum, te, ref)
rep("lg_prev + 0.35*elo     (spec mechanism, prior-season mean)  [REF]", ref, te, ref)
rep("lg_prev + 0.28*elo     (slope change only)", te.lg_prev + 0.28 * te.elo_sum, te, ref)
rep("lg_blend128 + 0.35*elo (blend only)", te.lg_blend + 0.35 * te.elo_sum, te, ref)
rep("lg_blend128 + 0.28*elo (expert's recommendation)", te.lg_blend + 0.28 * te.elo_sum, te, ref)
print("  per-season bias of lg_prev+0.35 vs lg_blend+0.28:")
for Y in (2022, 2023, 2024, 2025):
    x = te[te.season == Y]
    print(f"    {Y}: lg_prev={x.lg_prev.iloc[0]:.2f} realized={x.total_pts.mean():.2f} | bias lg_prev+.35={(x.lg_prev + .35*x.elo_sum - x.total_pts).mean():+.2f} lg_blend+.28={(x.lg_blend + .28*x.elo_sum - x.total_pts).mean():+.2f} 46.0+.35={(46 + .35*x.elo_sum - x.total_pts).mean():+.2f}")

print("\n== slope grid OOS (level = lg_blend128), paired vs b=0.28 ==")
refb = te.lg_blend + 0.28 * te.elo_sum
for b in (0.0, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50):
    rep(f"lg_blend + {b:.2f}*elo", te.lg_blend + b * te.elo_sum, te, refb)

print("\n== K_lg sensitivity, V1-type formula lg_blend(K) + 0.28*elo, seasons 2010-2025 (no fitting), paired vs lg_prev ==")
for K in (16, 32, 64, 128, 256, 512):
    LG = (K * d.lg_prev + d.n_before * d.lg_ytd.fillna(d.lg_prev)) / (K + d.n_before)
    for lab, s in [("2010-2021", d.season.between(2010, 2021)), ("2022-2025", d.test)]:
        x = d[s]; p = LG[s] + 0.28 * x.elo_sum; r = x.lg_prev + 0.28 * x.elo_sum
        dm, lo, hi, n = paired_mae_ci(p - x.total_pts, r - x.total_pts)
        print(f"  K={K:4d} {lab}: MAE={mae(p, x.total_pts):.3f} vs lg_prev {mae(r, x.total_pts):.3f}  d={dm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")
print("  -> in-season blend: check whether the gain concentrates in 2020-2023 (COVID / scoring drop) or is general")
for Y in range(2010, 2026):
    x = d[d.season == Y]; p = x.lg_blend + .28 * x.elo_sum; r = x.lg_prev + .28 * x.elo_sum
    print(f"    {Y}: lg_blend-lg_prev MAE diff={mae(p, x.total_pts) - mae(r, x.total_pts):+.3f}  |lg_prev-realized|={abs(x.lg_prev.iloc[0]-x.total_pts.mean()):.2f}")

print("\n== does the market residual load on elo_sum in ANY era (placebo for 'fully priced')? ==")
for lab, s in [("2009-2013", d.season.between(2009, 2013)), ("2014-2018", d.season.between(2014, 2018)), ("2019-2025", d.season.between(2019, 2025))]:
    f = smf.ols("total_err_mkt ~ elo_sum", data=d[s]).fit(cov_type="HC1")
    print(f"  {lab}: slope={f.params['elo_sum']:+.3f} (p={f.pvalues['elo_sum']:.2f}) n={int(s.sum())}")
