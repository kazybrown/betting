"""critic_04 (T4: EPA efficiency term on the market-free total -- the expert's one SUPPORTED finding).
Re-derives the numbers, then attacks: (1) season-level offsets in epa_sum from the lagged league
reference (2019's nflscrapR EPA mean is the reference for 2023's nflfastR EPA) vs BASE's per-season
bias; (2) the same rule with a leak-free season-to-date reference; (3) the 54.6% O/U claim vs the
BASE's own 52.3% and the always-over rate; (4) rolling-origin over ALL available seasons incl. the
modern ones (the totals critic found +0.003 for EPA in rolling-origin); (5) is it EPA or just the
8-game RECENCY window? swap in prior-8 points for/against; (6) robustness (Huber, per-season train
coefficients, blend version); (7) does the term move BASE toward the market (consistency argument)?
Fit 2009-2019, test 2023-2025 REG.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from critic_common import *  # noqa
pd.set_option("display.width", 220)

m = load_games_table()
m = add_std_centered(m)
d = combos(m, "r8d")
d["epa_sum_s"] = d.h_epa_off_r8s + d.a_epa_off_r8s + d.h_epa_def_r8s + d.a_epa_def_r8s
d["ppd_sum_s"] = d.h_ppd_r8s + d.a_ppd_r8s
d["pts_r8_sum"] = d.h_pts_r8d + d.a_pts_r8d + d.h_pts_allowed_r8d + d.a_pts_allowed_r8d      # prior-8 points for+against, both teams (league-relative)
d["pts_r8_sum_s"] = d.h_pts_r8s + d.a_pts_r8s + d.h_pts_allowed_r8s + d.a_pts_allowed_r8s
d["epa_sum_bl"] = d.h_epa_off_bld + d.a_epa_off_bld + d.h_epa_def_bld + d.a_epa_def_bld
d = reg_sample(d, BASE + ["lg_blend", "epa_sum", "spp_avg", "ppd_sum", "pts_r8_sum", "epa_sum_bl"])
tr, te = d[d.train], d[d.test]
print(f"sample: train n={len(tr)} test n={len(te)} (expert 2442 / 752)")

print("\n== (0) reproduce ==")
fb, pb = fit_pred(tr, te, BASE); res_b = te.total_pts.values - pb
f, p = fit_pred(tr, te, BASE + ["epa_sum"])
dm, lo, hi, n = paired_mae_ci(te.total_pts - p, res_b)
print(f"  BASE MAE={mae(pb, te.total_pts):.3f}; epa_sum train coef {f.params['epa_sum']:+.3f} (se {f.bse['epa_sum']:.3f}) p={f.pvalues['epa_sum']:.3f}; refit OOS dMAE {ci_str(dm, lo, hi)} (expert 4.051, -0.065 [-0.120,-0.014])")
rule = np.clip(4.0 * te.epa_sum.values, -3, 3); dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + rule), res_b)
w, l, pu = ou_rate(pb + rule, te.mkt_total, te.total_pts); w0, l0, pu0 = ou_rate(pb, te.mkt_total, te.total_pts)
adj = f.params["epa_sum"] * (te.epa_sum - tr.epa_sum.mean()); cal = ols(res_b, pd.DataFrame({"adj": adj}))
print(f"  rule clip(4*epa_sum, +/-3): dMAE {ci_str(dm, lo, hi)} dRMSE {rmse(pb+rule, te.total_pts)-rmse(pb, te.total_pts):+.3f}; O/U {w}-{l}-{pu} ({w/(w+l):.3f}) vs BASE {w0}-{l0}-{pu0} ({w0/(w0+l0):.3f}); calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})  (expert -0.051, 408-339, 0.68)")
fi = ols(te.total_pts - te.lg_blend, te[BASE + ["epa_sum"]]); print(f"  2023-25 in-sample coef {fi.params['epa_sum']:+.3f} (se {fi.bse['epa_sum']:.3f}) p={fi.pvalues['epa_sum']:.3f} (expert 3.73)")

print("\n== (1) season-level offsets in epa_sum (expert ref = prior-season league mean; 2019 nflscrapR mean used for 2023 nflfastR) ==")
tgf = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
lg = tgf[tgf.game_type == "REG"].groupby("season").epa_off.mean()
print("  league mean EPA/play by season: " + "  ".join(f"{y}:{v:+.3f}" for y, v in lg.items() if y >= 2017))
for y, x in te.groupby("season"):
    msk = te.season.values == y
    print(f"  {y}: mean epa_sum expert-ref {x.epa_sum.mean():+.3f} (std-ref {x.epa_sum_s.mean():+.3f}) -> mean rule adj {rule[msk].mean():+.2f} pts | BASE bias (act-pred) {np.mean(x.total_pts.values - pb[msk]):+.2f} | market bias {np.mean(x.total_pts - x.mkt_total):+.2f}")
print(by_season(te, te.total_pts - (pb + rule), res_b, "rule 4*epa_sum cap3 (expert ref)"))
print(by_season(te, te.total_pts - p, res_b, "refit BASE+epa_sum (expert ref)"))
rc = rule - rule.mean(); dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + rc), res_b)
print(f"  [diagnostic] rule with its pooled test mean ({rule.mean():+.2f}) removed: dMAE {ci_str(dm, lo, hi)}")
rcs = rule.copy()
for y in te.season.unique():
    msk = te.season.values == y; rcs[msk] = rule[msk] - rule[msk].mean()
dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + rcs), res_b)
print(f"  [diagnostic, leaky] rule demeaned within each test season (matchup shape only): dMAE {ci_str(dm, lo, hi)}")

print("\n== (2) leak-free season-to-date league reference (std ref) ==")
f_s, p_s = fit_pred(tr, te, BASE + ["epa_sum_s"]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p_s, res_b)
adj_s = f_s.params["epa_sum_s"] * (te.epa_sum_s - tr.epa_sum_s.mean()); cal = ols(res_b, pd.DataFrame({"adj": adj_s}))
print(f"  train coef {f_s.params['epa_sum_s']:+.3f} (se {f_s.bse['epa_sum_s']:.3f}) p={f_s.pvalues['epa_sum_s']:.3f}; corr(expert ref, std ref) test={np.corrcoef(te.epa_sum, te.epa_sum_s)[0,1]:.3f}; refit OOS dMAE {ci_str(dm, lo, hi)}; calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})")
rule_s = np.clip(4.0 * te.epa_sum_s.values, -3, 3); dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + rule_s), res_b)
w, l, pu = ou_rate(pb + rule_s, te.mkt_total, te.total_pts)
print(f"  rule clip(4*epa_sum_s, +/-3): dMAE {ci_str(dm, lo, hi)} dRMSE {rmse(pb+rule_s, te.total_pts)-rmse(pb, te.total_pts):+.3f}; mean adj {rule_s.mean():+.2f}; O/U {w}-{l}-{pu} ({w/(w+l):.3f})")
print(by_season(te, te.total_pts - (pb + rule_s), res_b, "rule 4*epa_sum cap3 (std ref)"))
fi = ols(te.total_pts - te.lg_blend, te[BASE + ["epa_sum_s"]]); print(f"  2023-25 in-sample coef (std ref) {fi.params['epa_sum_s']:+.3f} (se {fi.bse['epa_sum_s']:.3f}) p={fi.pvalues['epa_sum_s']:.3f}")

print("\n== (3) the O/U claim ==")
over = (te.total_pts > te.mkt_total).sum(); under = (te.total_pts < te.mkt_total).sum()
print(f"  2023-25 REG (n={len(te)}): overs {over}, unders {under}, always-over = {over/(over+under):.3f}; BASE (no EPA) already {w0}-{l0} ({w0/(w0+l0):.3f}) because BASE leans over (bias +0.63)")
pick_r = (pb + rule) > te.mkt_total.values; pick_b = pb > te.mkt_total.values; y_over = (te.total_pts > te.mkt_total).values; push = (te.total_pts == te.mkt_total).values
disc = (pick_r != pick_b) & ~push
r_win = ((pick_r == y_over) & disc).sum(); b_win = ((pick_b == y_over) & disc).sum()
print(f"  games where the EPA rule flips the O/U pick: {disc.sum()}; rule right {r_win}, BASE right {b_win} -> exact binomial p={stats.binomtest(int(r_win), int(r_win+b_win)).pvalue:.3f}")
print(f"  408-339 vs 50%: binomial p={stats.binomtest(408, 747).pvalue:.4f}; vs the always-over rate {over/(over+under):.3f}: p={stats.binomtest(408, 747, over/(over+under)).pvalue:.3f}")
sizes = np.abs(rule); big = sizes >= 1.0
w1, l1, _ = ou_rate((pb + rule)[big], te.mkt_total.values[big], te.total_pts.values[big])
print(f"  O/U when |adj|>=1 (n={big.sum()}): {w1}-{l1} ({w1/max(w1+l1,1):.3f}); agreement of rule's O/U pick with the market-vs-BASE direction: {np.mean(np.sign(rule) == np.sign(te.mkt_total.values - pb)):.3f}")

print("\n== (4) rolling-origin, all seasons (fit < Y; 2023 fits <= 2019; 2024 adds 2023; 2025 adds 2023-24) ==")
rows = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    a, b = d[d.season < Y], d[d.season == Y]
    f0, p0 = fit_pred(a, b, BASE)
    out = [Y, len(b), mae(p0, b.total_pts)]
    for c in ["epa_sum", "epa_sum_s", "pts_r8_sum_s", "ppd_sum_s"]:
        f1, p1 = fit_pred(a, b, BASE + [c]); out += [mae(p1, b.total_pts) - mae(p0, b.total_pts), f1.params[c]]
    out.append(mae(p0 + np.clip(4.0 * b.epa_sum_s.values, -3, 3), b.total_pts) - mae(p0, b.total_pts))
    rows.append(out)
r = pd.DataFrame(rows, columns=["Y", "n", "BASE", "d_epa", "b_epa", "d_epa_s", "b_epa_s", "d_pts8_s", "b_pts8_s", "d_ppd_s", "b_ppd_s", "d_rule4_s"])
print(r.round(3).to_string(index=False))
for c in ["d_epa", "d_epa_s", "d_pts8_s", "d_ppd_s", "d_rule4_s"]:
    print(f"  {c:10s} mean 2013-19 {r[r.Y<=2019][c].mean():+.3f} ({int((r[r.Y<=2019][c]<0).sum())}/7 better) | 2023-25 {r[r.Y>=2023][c].mean():+.3f} ({int((r[r.Y>=2023][c]<0).sum())}/3) | all 10 {r[c].mean():+.3f}")
# pooled paired CI over all rolling-origin test games
errs0, errs1, errs2, errs3 = [], [], [], []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    a, b = d[d.season < Y], d[d.season == Y]
    f0, p0 = fit_pred(a, b, BASE); f1, p1 = fit_pred(a, b, BASE + ["epa_sum_s"]); f3, p3 = fit_pred(a, b, BASE + ["pts_r8_sum_s"])
    errs0 += list(b.total_pts.values - p0); errs1 += list(b.total_pts.values - p1); errs2 += list(b.total_pts.values - (p0 + np.clip(4.0 * b.epa_sum_s.values, -3, 3))); errs3 += list(b.total_pts.values - p3)
for lab, e in [("refit +epa_sum_s", errs1), ("fixed rule 4*epa_s cap3", errs2), ("refit +pts_r8_sum_s (recency placebo)", errs3)]:
    dm, lo, hi, n = paired_mae_ci(e, errs0); print(f"  pooled rolling-origin {lab:38s} dMAE vs BASE {ci_str(dm, lo, hi)} n={n}")

print("\n== (5) EPA or just an 8-game recency window? prior-8 POINTS for+against as the extra term (fit train, OOS 2023-25) ==")
for c in ["epa_sum", "epa_sum_s", "pts_r8_sum", "pts_r8_sum_s", "ppd_sum", "ppd_sum_s", "epa_sum_bl"]:
    f1, p1 = fit_pred(tr, te, BASE + [c]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p1, res_b)
    fi = ols(te.total_pts - te.lg_blend, te[BASE + [c]])
    adj = f1.params[c] * (te[c] - tr[c].mean()); cal = ols(res_b, pd.DataFrame({"adj": adj}))
    print(f"  + {c:13s} train {f1.params[c]:+.3f} (p={f1.pvalues[c]:.3f}; per SD {f1.params[c]*tr[c].std():+.2f}) | 23-25 in-sample {fi.params[c]:+.3f} (p={fi.pvalues[c]:.3f}) | OOS dMAE {ci_str(dm, lo, hi)} | calib {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})")
f1, p1 = fit_pred(tr, te, BASE + ["epa_sum_s", "pts_r8_sum_s"]); dm, lo, hi, n = paired_mae_ci(te.total_pts - p1, res_b)
print(f"  + both epa_sum_s and pts_r8_sum_s: epa {f1.params['epa_sum_s']:+.3f} (p={f1.pvalues['epa_sum_s']:.3f}), pts8 {f1.params['pts_r8_sum_s']:+.3f} (p={f1.pvalues['pts_r8_sum_s']:.3f}) | OOS dMAE {ci_str(dm, lo, hi)}")

print("\n== (6) robustness of the train coefficient ==")
h = sm.RLM((tr.total_pts - tr.lg_blend).values, sm.add_constant(tr[BASE + ["epa_sum_s"]].astype(float)), M=sm.robust.norms.HuberT()).fit()
print(f"  Huber 2009-19: epa_sum_s {h.params['epa_sum_s']:+.3f} (se {h.bse['epa_sum_s']:.3f})")
rows = []
for y, x in d.groupby("season"):
    f = ols(x.total_pts - x.lg_blend, x[BASE + ["epa_sum_s"]]); rows.append((y, f.params["epa_sum_s"], f.bse["epa_sum_s"]))
print("  single-season coefs (BASE + epa_sum_s): " + "  ".join(f"{y}:{c:+.1f}({s:.1f})" for y, c, s in rows))
print(f"  seasons with positive coef: {sum(c>0 for _, c, _ in rows)}/{len(rows)}; inverse-variance pooled {sum(c/s**2 for _, c, s in rows)/sum(1/s**2 for _, c, s in rows):+.2f} (se {np.sqrt(1/sum(1/s**2 for _, c, s in rows)):.2f})")

print("\n== (7) market-consistency: does the EPA term move BASE toward the market total? and the market's own EPA weight ==")
for lab, pp in [("BASE", pb), ("BASE + refit epa_sum_s", p_s), ("BASE + rule 4*epa_s cap3", pb + rule_s)]:
    print(f"  {lab:28s} MAE vs MARKET total = {mae(pp, te.mkt_total):.3f}; MAE vs result = {mae(pp, te.total_pts):.3f}")
for lab, x in [("train", tr), ("test", te)]:
    fm = ols(x.mkt_total - x.lg_blend, x[BASE + ["epa_sum_s"]]); fr = ols(x.total_pts - x.lg_blend, x[BASE + ["epa_sum_s"]])
    print(f"  {lab}: market slope on epa_sum_s {fm.params['epa_sum_s']:+.2f} (se {fm.bse['epa_sum_s']:.2f}) | realized slope {fr.params['epa_sum_s']:+.2f} (se {fr.bse['epa_sum_s']:.2f}) | residual-vs-market slope {ols(x.total_err_mkt, x[['epa_sum_s']]).params['epa_sum_s']:+.2f} (p={ols(x.total_err_mkt, x[['epa_sum_s']]).pvalues['epa_sum_s']:.2f})")
