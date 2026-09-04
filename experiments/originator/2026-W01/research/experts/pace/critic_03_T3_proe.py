"""critic_03 (T3 PROE / pass rate -> totals). Re-derives the expert's numbers, then attacks: (1) the
season-level offsets in the expert's league-relative PROE (its xpass logit was fit on 2009-19 and the
league reference lags a season: 2019's +0.70 mean is the reference for 2023 whose mean is -0.13);
(2) the same tests with nflfastR's OWN pass_oe (official xpass model, prior-8, 2023-25) as the PROE
series; (3) rolling-origin vs market; (4) PROE on the rating-only (Elo) total that ORIGINATOR
actually publishes, where the market-free coefficient is much larger; (5) placebo cost band.
Fit 2009-2019, test 2023-2025 REG.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from critic_common import *  # noqa
pd.set_option("display.width", 220)

m = load_games_table()
m = add_std_centered(m)
# nflfastR pass_oe prior-8 (2023-25 only), never crossing seasons' era, league-relative via season-to-date mean
tg = pd.read_csv(HERE / "_teamgame.csv", low_memory=False)
tg = tg[tg.season >= 2023].sort_values(["team", "season", "gameday", "gid"])
tg["poe_r8"] = tg.groupby("team").proe_fastr.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
tg["proe_r8_mine"] = tg.groupby("team").proe.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
sub = tg[["gid", "team", "poe_r8", "proe_r8_mine"]]
m = m.merge(sub.rename(columns={"team": "home", "poe_r8": "h_poe_r8", "proe_r8_mine": "h_proe_r8m"}), on=["gid", "home"], how="left")
m = m.merge(sub.rename(columns={"team": "away", "poe_r8": "a_poe_r8", "proe_r8_mine": "a_proe_r8m"}), on=["gid", "away"], how="left")
d = combos(m, "r8d")
d["proe_sum_s"] = d.h_proe_r8s + d.a_proe_r8s
d["pr_sum_s"] = d.h_pr_neut_r8s + d.a_pr_neut_r8s
d["poe_sum"] = d.h_poe_r8 + d.a_poe_r8
d = reg_sample(d, BASE + ["lg_blend", "proe_sum", "pr_sum", "proe_sum_s"])
tr, te = d[d.train], d[d.test]
print(f"sample: train n={len(tr)} test n={len(te)} (expert 2442 / 752)")

print("\n== (0) reproduce ==")
f1 = ols(tr.total_err_mkt, tr[["proe_sum"]]); f2 = ols(te.total_err_mkt, te[["proe_sum"]])
adj = f1.params["const"] + f1.params["proe_sum"] * te.proe_sum
dm, lo, hi, n = paired_mae_ci(te.total_pts - (te.mkt_total + adj), te.total_err_mkt)
print(f"  market residual ~ proe_sum: train {f1.params['proe_sum']:+.3f} (se {f1.bse['proe_sum']:.3f}) p={f1.pvalues['proe_sum']:.3f} | test {f2.params['proe_sum']:+.3f} (se {f2.bse['proe_sum']:.3f}) p={f2.pvalues['proe_sum']:.3f} | OOS dMAE {ci_str(dm, lo, hi)}  (expert +0.044/+0.022, +0.001)")
fb, pb = fit_pred(tr, te, BASE); res_b = te.total_pts.values - pb
f, p = fit_pred(tr, te, BASE + ["proe_sum"]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p, res_b)
adjp = f.params["proe_sum"] * (te.proe_sum - tr.proe_sum.mean()); cal = ols(res_b, pd.DataFrame({"adj": adjp}))
print(f"  market-free BASE + proe_sum: train coef {f.params['proe_sum']:+.3f} (se {f.bse['proe_sum']:.3f}) | OOS dMAE {ci_str(dm, lo, hi)} | calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})  (expert +0.170, +0.018, 0.44)")

print("\n== (1) season-level offsets in the expert's league-relative PROE ==")
tgf = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
lg = tgf[tgf.game_type == "REG"].groupby("season").proe.mean()
print("  league mean PROE (expert xpass) by season: " + "  ".join(f"{y}:{v:+.2f}" for y, v in lg.items() if y >= 2017))
for y, x in te.groupby("season"):
    print(f"  {y}: mean proe_sum expert-ref {x.proe_sum.mean():+.2f} | std-ref {x.proe_sum_s.mean():+.2f} | mean train-fit adj {adjp[te.season==y].mean():+.2f} pts | BASE bias(act-pred) {np.mean(x.total_pts.values - pb[te.season.values==y]):+.2f}")
print(by_season(te, te.total_pts - p, res_b, "BASE+proe_sum (expert ref)"))
f_s, p_s = fit_pred(tr, te, BASE + ["proe_sum_s"]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p_s, res_b)
adjs = f_s.params["proe_sum_s"] * (te.proe_sum_s - tr.proe_sum_s.mean()); cal = ols(res_b, pd.DataFrame({"adj": adjs}))
w, l, pu = ou_rate(p_s, te.mkt_total, te.total_pts)
print(f"  std ref: train coef {f_s.params['proe_sum_s']:+.3f} (se {f_s.bse['proe_sum_s']:.3f}) | OOS dMAE {ci_str(dm, lo, hi)} | calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f}) | O/U {w}-{l}-{pu} ({w/(w+l):.3f})")
print(by_season(te, te.total_pts - p_s, res_b, "BASE+proe_sum (std ref)"))
f_te = ols(te.total_pts - te.lg_blend, te[BASE + ["proe_sum_s"]])
print(f"  2023-25 in-sample coef (std ref): {f_te.params['proe_sum_s']:+.3f} (se {f_te.bse['proe_sum_s']:.3f}) p={f_te.pvalues['proe_sum_s']:.3f}")
f_m1 = ols(te.total_err_mkt, te[["proe_sum_s"]])
print(f"  vs market, test, std ref: {f_m1.params['proe_sum_s']:+.3f} (se {f_m1.bse['proe_sum_s']:.3f}) p={f_m1.pvalues['proe_sum_s']:.3f}")

print("\n== (2) nflfastR's own pass_oe (prior-8, 2023-25) as the PROE series ==")
tp = te.dropna(subset=["poe_sum"]).copy()
print(f"  n={len(tp)}; corr(expert proe_sum r8, fastR poe_sum r8) = {np.corrcoef(tp.proe_sum, tp.poe_sum)[0,1]:.3f}; SD poe_sum={tp.poe_sum.std():.2f} (expert proe_sum SD {tp.proe_sum.std():.2f})")
f = ols(tp.total_err_mkt, tp[["poe_sum"]]); print(f"  market residual ~ fastR poe_sum (2023-25 in-sample): {f.params['poe_sum']:+.3f} (se {f.bse['poe_sum']:.3f}) p={f.pvalues['poe_sum']:.3f} [per SD {f.params['poe_sum']*tp.poe_sum.std():+.2f}]")
f = ols(tp.total_pts - tp.lg_blend, tp[BASE + ["poe_sum"]]); print(f"  market-free BASE + fastR poe_sum (2023-25 in-sample): {f.params['poe_sum']:+.3f} (se {f.bse['poe_sum']:.3f}) p={f.pvalues['poe_sum']:.3f} [per SD {f.params['poe_sum']*tp.poe_sum.std():+.2f}]")
f = ols(tp.mkt_total - tp.lg_blend, tp[BASE + ["poe_sum"]]); print(f"  MARKET total ~ BASE + fastR poe_sum (2023-25): {f.params['poe_sum']:+.3f} (se {f.bse['poe_sum']:.3f}) p={f.pvalues['poe_sum']:.3f}  <- the market's own weight on PROE")
# apply the expert's train-fit coefficient (0.170) to the fastR series: calibration
pb_p = pb[te.poe_sum.notna().values]; adjf = 0.170 * (tp.poe_sum - tp.poe_sum.mean())
cal = ols(tp.total_pts.values - pb_p, pd.DataFrame({"adj": adjf}))
dm, lo, hi, n = paired_mae_ci(tp.total_pts.values - (pb_p + adjf), tp.total_pts.values - pb_p)
print(f"  train-fit 0.170 x fastR poe_sum (centered) on BASE: calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f}); dMAE {ci_str(dm, lo, hi)}")

print("\n== (3) rolling-origin vs MARKET (fit < Y, market + train-fit adj), std ref ==")
acc = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    a, b = d[d.season < Y], d[d.season == Y]
    f = ols(a.total_err_mkt, a[["proe_sum_s"]]); adj = f.params["const"] + f.params["proe_sum_s"] * b.proe_sum_s
    acc.append((Y, mae(b.mkt_total + adj, b.total_pts) - mae(b.mkt_total, b.total_pts), f.params["proe_sum_s"]))
print("  " + "  ".join(f"{y}:{v:+.3f}(b={c:+.3f})" for y, v, c in acc) + f" | mean {np.mean([v for _, v, _ in acc]):+.3f}, better {sum(v<0 for _, v, _ in acc)}/{len(acc)}")

print("\n== (4) PROE on the RATING-ONLY total (Elo + QB + env; no pf/pa) -- what ORIGINATOR publishes ==")
E = ["elo_sum", "qb_sum", "dome", "wind_f", "div"]
fe, pe = fit_pred(tr, te, E); rese = te.total_pts.values - pe
for lab, c in [("expert ref", "proe_sum"), ("std ref", "proe_sum_s")]:
    f, p = fit_pred(tr, te, E + [c]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p, rese)
    dmm, lom, him, _ = paired_mae_ci(te.total_pts - p, te.total_err_mkt)
    adj = f.params[c] * (te[c] - tr[c].mean()); cal = ols(rese, pd.DataFrame({"adj": adj}))
    fi = ols(te.total_pts - te.lg_blend, te[E + [c]])
    print(f"  Elo-base MAE={mae(pe, te.total_pts):.3f} + {c:11s} [{lab}]: train coef {f.params[c]:+.3f} (p={f.pvalues[c]:.3f}; per SD {f.params[c]*tr[c].std():+.2f}) | 2023-25 in-sample {fi.params[c]:+.3f} (p={fi.pvalues[c]:.3f}) | OOS dMAE {ci_str(dm, lo, hi)} | vs mkt {ci_str(dmm, lom, him)} | calib {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})")
    print(by_season(te, te.total_pts - p, rese, f"Elo-base + {c}"))
acc = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    a, b = d[d.season < Y], d[d.season == Y]
    f0, p0 = fit_pred(a, b, E); f1_, p1 = fit_pred(a, b, E + ["proe_sum_s"])
    acc.append((Y, mae(p1, b.total_pts) - mae(p0, b.total_pts)))
print("  rolling-origin Elo-base + proe_sum_s minus Elo-base: " + "  ".join(f"{y}:{v:+.3f}" for y, v in acc) + f" | mean 2013-19 {np.mean([v for y, v in acc if y<=2019]):+.3f}, 2023-25 {np.mean([v for y, v in acc if y>=2023]):+.3f}")
# how much closer to the market does PROE bring the rating-only total?
f, p = fit_pred(tr, te, E + ["proe_sum_s"])
print(f"  distance to MARKET total: Elo-base MAE vs mkt={mae(pe, te.mkt_total):.3f}; Elo-base+PROE={mae(p, te.mkt_total):.3f}; LEAN={mae(pb, te.mkt_total):.3f}")

print("\n== (5) placebo band: cost of a PROE-sized adjustment (mean|adj| ~0.9 pts) carrying no signal, on LEAN BASE (300 within-season shuffles) ==")
rng = np.random.default_rng(2); sims = []
real = mae(p_s, te.total_pts) - mae(pb, te.total_pts)
for _ in range(300):
    x = te.proe_sum_s.values.copy()
    for y, idx in te.groupby("season").indices.items():
        x[idx] = x[rng.permutation(idx)]
    sims.append(mae(pb + f_s.params["proe_sum_s"] * (x - tr.proe_sum_s.mean()), te.total_pts) - mae(pb, te.total_pts))
sims = np.array(sims)
print(f"  real dMAE={real:+.3f} | placebo mean={sims.mean():+.3f} sd={sims.std():.3f} [2.5%,97.5%]=[{np.percentile(sims,2.5):+.3f},{np.percentile(sims,97.5):+.3f}] | P(placebo <= real)={np.mean(sims <= real):.3f}")
