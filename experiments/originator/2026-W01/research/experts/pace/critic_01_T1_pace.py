"""critic_01 (T1 pace -> totals). Re-derives the expert's key numbers from the expert's own feature
table, then attacks: (1) season-level offsets in the 'league-relative' pace feature caused by the
lagged league reference, and the per-season bias of the market-free BASE; (2) the same rules with a
leak-free season-to-date league reference; (3) a placebo (pace shuffled across games within season)
to see how much of the 'harm' of the spec rule is just the cost of any +/-0.5-pt noise; (4) pace on
top of the ORIGINATOR-style rating-only total (spec: lg_prior + 0.35*elo_sum; and a fitted Elo-only
model) -- LEAN's pf/pa may already absorb pace; (5) vs market: pooled robust (Huber) and rolling-
origin; (6) power of the 752-game test and the realized-pace upper bound; (7) independent check of
the 'dispersion fell' claim with the totals expert's sec_per_play definition.
Fit 2009-2019, test 2023-2025 REG (as the expert). Rolling-origin labelled.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from critic_common import *  # noqa
pd.set_option("display.width", 220)

m0 = load_games_table()
m0 = add_std_centered(m0)
V = "r8d"
d = combos(m0, V)
d = reg_sample(d, BASE + ["lg_blend", "spp_avg", "sppr_avg", "gplays_avg", "ppd_sum", "epa_sum"])
tr, te = d[d.train], d[d.test]
print(f"sample: train 2009-19 n={len(tr)} | test 2023-25 n={len(te)}  (expert: 2442 / 752)")
print(f"sign check: corr(mkt_spread, margin)={np.corrcoef(d.mkt_spread, d.margin)[0,1]:+.3f}; corr(spp_avg, realized game plays) train={np.corrcoef(tr.spp_avg, tr.h_plays_act+tr.a_plays_act)[0,1]:+.3f}")

# ---------- (0) reproduce ----------
print("\n== (0) reproduce the expert's headline numbers ==")
fb, pb = fit_pred(tr, te, BASE); res_b = te.total_pts.values - pb
f1, _ = fit_pred(tr, te, BASE + ["spp_avg"])
print(f"  BASE test MAE={mae(pb, te.total_pts):.3f} (expert 10.290) bias(actual-pred)={np.mean(res_b):+.2f} (expert +0.63); market MAE={mae(te.mkt_total, te.total_pts):.3f}")
print(f"  spp_avg train coef {f1.params['spp_avg']:+.3f} (se {f1.bse['spp_avg']:.3f}) p={f1.pvalues['spp_avg']:.3f}  (expert -0.541, se 0.215, p=0.012)")
f2 = ols(te.total_pts - te.lg_blend, te[BASE + ["spp_avg"]])
print(f"  spp_avg 2023-25 in-sample coef {f2.params['spp_avg']:+.3f} (se {f2.bse['spp_avg']:.3f}) p={f2.pvalues['spp_avg']:.3f}  (expert +0.132, se 0.516)")
adj_lin = f1.params["spp_avg"] * (te.spp_avg - tr.spp_avg.mean())
cal = ols(res_b, pd.DataFrame({"adj": adj_lin}))
print(f"  calibration slope of train-fit adj: {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})  (expert -0.27, se 0.91)")
lo_q, hi_q = np.percentile(pd.concat([tr[f"h_spp_neut_{V}"], tr[f"a_spp_neut_{V}"]]), [25, 75])


def spec_rule(fr, hcol, acol, lo_q, hi_q, size=1.0):
    def cls(x): return np.where(x <= lo_q, "F", np.where(x >= hi_q, "S", "M"))
    b = pd.Series(cls(fr[hcol])).str.cat(pd.Series(cls(fr[acol]))).values
    return size * np.select([b == "FF", b == "SS", np.isin(b, ["FM", "MF"]), np.isin(b, ["SM", "MS"])], [1.5, -1.5, 0.75, -0.75], 0.0)


rules = {"spec bucket (+/-1.5 FF/SS, +/-0.75 mixed)": spec_rule(te, f"h_spp_neut_{V}", f"a_spp_neut_{V}", lo_q, hi_q),
         "spec x0.5": spec_rule(te, f"h_spp_neut_{V}", f"a_spp_neut_{V}", lo_q, hi_q, 0.5),
         "linear -0.5*spp_avg cap 1": np.clip(-0.5 * te.spp_avg.values, -1, 1),
         "linear -0.25*spp_avg cap 0.5": np.clip(-0.25 * te.spp_avg.values, -0.5, 0.5)}
for lab, adj in rules.items():
    dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + adj), res_b)
    print(f"  {lab:46s} dMAE vs BASE {ci_str(dm, lo, hi)}  mean adj={adj.mean():+.2f} mean|adj|={np.abs(adj).mean():.2f}")

# ---------- (1) season-level offsets ----------
print("\n== (1) season-level offsets: expert's league-relative spp (prior-season ref) vs the season's own level; BASE bias by season ==")
tgf = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
lg = tgf[tgf.game_type == "REG"].groupby("season").spp_neut.mean()
print("  league mean neutral sec/play by season: " + "  ".join(f"{y}:{v:.2f}" for y, v in lg.items() if y >= 2017))
rows = []
for y, x in te.groupby("season"):
    rows.append((y, len(x), x.spp_avg.mean(), (x[["h_spp_neut_r8s", "a_spp_neut_r8s"]].mean(axis=1)).mean(), np.mean(x.total_pts.values - pb[te.season.values == y]),
                 np.mean(x.total_pts - x.mkt_total), rules["spec bucket (+/-1.5 FF/SS, +/-0.75 mixed)"][te.season.values == y].mean(),
                 rules["linear -0.5*spp_avg cap 1"][te.season.values == y].mean()))
print(pd.DataFrame(rows, columns=["season", "n", "mean spp_avg (expert ref)", "mean spp_avg (std ref)", "BASE bias (act-pred)", "market bias", "mean spec adj", "mean lin adj"]).round(3).to_string(index=False))
print("  -> in 2024 every team is scored ~+0.8 s 'slow' against the 2023 reference, so the rules subtract points in the season where BASE was already too LOW.")
for lab, adj in rules.items():
    print(by_season(te, te.total_pts - (pb + adj), res_b, lab))

# ---------- (2) alternative leak-free reference ----------
print("\n== (2) same rules with a leak-free SEASON-TO-DATE league reference (offsets removed) ==")
d2 = d.copy(); d2["spp_avg_s"] = (d2.h_spp_neut_r8s + d2.a_spp_neut_r8s) / 2
tr2, te2 = d2[d2.train], d2[d2.test]
f1s, _ = fit_pred(tr2, te2, BASE + ["spp_avg_s"])
print(f"  train coef on std-ref spp_avg: {f1s.params['spp_avg_s']:+.3f} (se {f1s.bse['spp_avg_s']:.3f}) p={f1s.pvalues['spp_avg_s']:.3f}; corr(expert ref, std ref) test={np.corrcoef(te2.spp_avg, te2.spp_avg_s)[0,1]:.3f}")
f2s = ols(te2.total_pts - te2.lg_blend, te2[BASE + ["spp_avg_s"]])
print(f"  2023-25 in-sample coef: {f2s.params['spp_avg_s']:+.3f} (se {f2s.bse['spp_avg_s']:.3f}) p={f2s.pvalues['spp_avg_s']:.3f}")
adj_s = f1s.params["spp_avg_s"] * (te2.spp_avg_s - tr2.spp_avg_s.mean()); cal = ols(res_b, pd.DataFrame({"adj": adj_s}))
print(f"  calibration slope (std ref): {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f})")
lo_s, hi_s = np.percentile(pd.concat([tr2.h_spp_neut_r8s, tr2.a_spp_neut_r8s]), [25, 75])
rules_s = {"spec bucket, std ref": spec_rule(te2, "h_spp_neut_r8s", "a_spp_neut_r8s", lo_s, hi_s),
           "spec x0.5, std ref": spec_rule(te2, "h_spp_neut_r8s", "a_spp_neut_r8s", lo_s, hi_s, 0.5),
           "linear -0.5*spp cap 1, std ref": np.clip(-0.5 * te2.spp_avg_s.values, -1, 1),
           "linear -0.25*spp cap 0.5, std ref": np.clip(-0.25 * te2.spp_avg_s.values, -0.5, 0.5),
           "linear train-fit b*spp, std ref": adj_s.values}
for lab, adj in rules_s.items():
    dm, lo, hi, n = paired_mae_ci(te2.total_pts - (pb + adj), res_b)
    w, l, pu = ou_rate(pb + adj, te2.mkt_total, te2.total_pts)
    print(f"  {lab:46s} dMAE vs BASE {ci_str(dm, lo, hi)}  dRMSE {rmse(pb+adj, te2.total_pts)-rmse(pb, te2.total_pts):+.3f} mean adj={adj.mean():+.2f}  O/U {w}-{l}-{pu} ({w/(w+l):.3f})")
    print(by_season(te2, te2.total_pts - (pb + adj), res_b, lab))
# also: rule demeaned within the test set (diagnostic, uses test-set mean -> mildly leaky; isolates the shape of the rule)
adj = rules["spec bucket (+/-1.5 FF/SS, +/-0.75 mixed)"]; adjc = adj - adj.mean()
dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + adjc), res_b)
print(f"  [diagnostic] spec rule with its test-set mean removed (shape only): dMAE {ci_str(dm, lo, hi)}")

# ---------- (3) placebo ----------
print("\n== (3) placebo: shuffle both teams' prior pace across games WITHIN season (500 draws); cost of a random rule of the same size ==")
rng = np.random.default_rng(1)
real = {k: mae(pb + v, te.total_pts) - mae(pb, te.total_pts) for k, v in rules.items()}
sims = {k: [] for k in rules}
for _ in range(500):
    tp = te.copy()
    for y, idx in te.groupby("season").indices.items():
        perm = rng.permutation(idx)
        for c in (f"h_spp_neut_{V}", f"a_spp_neut_{V}"):
            tp.iloc[idx, tp.columns.get_loc(c)] = te[c].values[perm]
    tp["spp_avg"] = (tp[f"h_spp_neut_{V}"] + tp[f"a_spp_neut_{V}"]) / 2
    pr = {"spec bucket (+/-1.5 FF/SS, +/-0.75 mixed)": spec_rule(tp, f"h_spp_neut_{V}", f"a_spp_neut_{V}", lo_q, hi_q),
          "spec x0.5": spec_rule(tp, f"h_spp_neut_{V}", f"a_spp_neut_{V}", lo_q, hi_q, 0.5),
          "linear -0.5*spp_avg cap 1": np.clip(-0.5 * tp.spp_avg.values, -1, 1),
          "linear -0.25*spp_avg cap 0.5": np.clip(-0.25 * tp.spp_avg.values, -0.5, 0.5)}
    for k in rules:
        sims[k].append(mae(pb + pr[k], te.total_pts) - mae(pb, te.total_pts))
for k in rules:
    s = np.array(sims[k])
    print(f"  {k:46s} real dMAE={real[k]:+.3f} | placebo mean={s.mean():+.3f} sd={s.std():.3f} 2.5/97.5%=[{np.percentile(s,2.5):+.3f},{np.percentile(s,97.5):+.3f}] | P(placebo <= real)={np.mean(s <= real[k]):.3f}")
print("  (a placebo rule with the same mean offset also 'hurts'; the real rule must beat the placebo band to show anti-signal)")

# ---------- (4) pace on rating-only totals (what ORIGINATOR actually adds pace to) ----------
print("\n== (4) pace on top of RATING-ONLY totals (no pf/pa): spec-style lg_prior + 0.35*elo_sum (no fit) and fitted Elo(+env) models ==")
d["orig_spec"] = d.lg_blend + 0.35 * d.elo_sum
tr, te = d[d.train], d[d.test]
bases = {"spec: lg_blend + 0.35*elo_sum (no fitting)": (None, te.orig_spec.values),
         "fit: lg_blend + b*elo_sum": fit_pred(tr, te, ["elo_sum"]),
         "fit: elo_sum + qb_sum + dome + wind + div": fit_pred(tr, te, ["elo_sum", "qb_sum", "dome", "wind_f", "div"]),
         "fit: LEAN (elo+pf+pa+qb+env)": fit_pred(tr, te, BASE)}
for lab, (f, p0) in bases.items():
    r0 = te.total_pts.values - p0
    out = [f"  {lab:44s} MAE={mae(p0, te.total_pts):.3f} bias={np.mean(r0):+.2f}"]
    for rl, adj in [("spec rule", rules["spec bucket (+/-1.5 FF/SS, +/-0.75 mixed)"]), ("lin -0.5 cap1", rules["linear -0.5*spp_avg cap 1"]), ("lin -0.5 cap1 std-ref", rules_s["linear -0.5*spp cap 1, std ref"])]:
        dm, lo, hi, n = paired_mae_ci(te.total_pts - (p0 + adj), r0)
        out.append(f"{rl}: {ci_str(dm, lo, hi)}")
    # fitted pace coefficient on that base (train) and in-sample test
    cols = {"spec: lg_blend + 0.35*elo_sum (no fitting)": ["elo_sum"], "fit: lg_blend + b*elo_sum": ["elo_sum"],
            "fit: elo_sum + qb_sum + dome + wind + div": ["elo_sum", "qb_sum", "dome", "wind_f", "div"], "fit: LEAN (elo+pf+pa+qb+env)": BASE}[lab]
    ftr = ols(tr.total_pts - tr.lg_blend, tr[cols + ["spp_avg"]]); fte = ols(te.total_pts - te.lg_blend, te[cols + ["spp_avg"]])
    out.append(f"spp coef train {ftr.params['spp_avg']:+.3f}(p={ftr.pvalues['spp_avg']:.3f}) test-in-sample {fte.params['spp_avg']:+.3f}(p={fte.pvalues['spp_avg']:.3f})")
    print(" | ".join(out))
# rolling-origin on the Elo-only base, all seasons incl. modern
print("  rolling-origin (fit < Y; 2023 fit<=2019) Elo-only base vs +spp_avg (std ref), MAE:")
acc = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    trY, teY = d2[d2.season < Y], d2[d2.season == Y]
    f0, p0 = fit_pred(trY, teY, ["elo_sum", "qb_sum", "dome", "wind_f", "div"]); f1_, p1 = fit_pred(trY, teY, ["elo_sum", "qb_sum", "dome", "wind_f", "div", "spp_avg_s"])
    acc.append((Y, len(teY), mae(p0, teY.total_pts), mae(p1, teY.total_pts), f1_.params["spp_avg_s"]))
r = pd.DataFrame(acc, columns=["Y", "n", "elo_base", "+spp_std", "coef"]); r["d"] = r["+spp_std"] - r.elo_base
print("   " + "  ".join(f"{int(y)}:{dd:+.3f}(b={c:+.2f})" for y, dd, c in zip(r.Y, r.d, r.coef)) + f"  | mean 2013-19 {r[r.Y<=2019].d.mean():+.3f}, mean 2023-25 {r[r.Y>=2023].d.mean():+.3f}")

# ---------- (5) vs market ----------
print("\n== (5) vs MARKET close: pooled robust regression and rolling-origin ==")
allp = d2.copy()
for c, lab in [("spp_avg", "expert ref"), ("spp_avg_s", "std ref")]:
    h = sm.RLM(allp.total_err_mkt.values, sm.add_constant(allp[[c]].astype(float)), M=sm.robust.norms.HuberT()).fit()
    o = ols(allp.total_err_mkt, allp[[c]])
    print(f"  pooled 2009-19+2023-25 n={len(allp)}: (total - mkt) ~ {c} [{lab}]: OLS {o.params[c]:+.3f} (se {o.bse[c]:.3f}, p={o.pvalues[c]:.3f}) | Huber {h.params[c]:+.3f} (se {h.bse[c]:.3f})")
acc = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    trY, teY = allp[allp.season < Y], allp[allp.season == Y]
    f = ols(trY.total_err_mkt, trY[["spp_avg_s"]]); adj = f.params["const"] + f.params["spp_avg_s"] * teY.spp_avg_s
    acc.append((Y, len(teY), mae(teY.mkt_total + adj, teY.total_pts) - mae(teY.mkt_total, teY.total_pts), f.params["spp_avg_s"]))
r = pd.DataFrame(acc, columns=["Y", "n", "dMAE", "coef"])
print("  rolling-origin market + train-fit spp adj (std ref): " + "  ".join(f"{int(y)}:{dd:+.3f}" for y, dd in zip(r.Y, r.dMAE)) + f" | mean {r.dMAE.mean():+.3f}, better {int((r.dMAE<0).sum())}/{len(r)}")
# market's own pace weighting, std ref
for lab, x in [("train", tr2), ("test", te2)]:
    f = ols(x.mkt_total - x.lg_blend, x[BASE + ["spp_avg_s"]])
    print(f"  market total ~ BASE + spp_avg_s [{lab}]: {f.params['spp_avg_s']:+.3f} (se {f.bse['spp_avg_s']:.3f}, p={f.pvalues['spp_avg_s']:.3f})")

# ---------- (6) power, upper bound ----------
print("\n== (6) power and upper bound ==")
f_te = ols(te.total_err_mkt, te[["spp_avg"]])
print(f"  test-era residual-vs-market coefficient se = {f_te.bse['spp_avg']:.3f} pts/s -> a train-sized effect (-0.30) has power ~{stats.norm.sf(1.96 - 0.30/f_te.bse['spp_avg']):.2f}; minimum detectable |effect| at 80% power ~{2.8*f_te.bse['spp_avg']:.2f} pts/s")
f_teb = ols(te.total_pts - te.lg_blend, te[BASE + ["spp_avg"]])
print(f"  market-free 2023-25 in-sample se = {f_teb.bse['spp_avg']:.3f}: the 2009-19 coefficient (-0.54) is {abs(-0.54 - f_teb.params['spp_avg'])/f_teb.bse['spp_avg']:.2f} se from the 2023-25 estimate -> 'the effect vanished' is NOT statistically established, only 'not detected'")
gp = te.h_plays_act + te.a_plays_act
print(f"  realized game plays vs market residual (test): corr={np.corrcoef(gp, te.total_err_mkt)[0,1]:+.3f}; prior spp_avg -> realized plays R^2 = {np.corrcoef(te.spp_avg, gp)[0,1]**2:.3f} (train {np.corrcoef(tr.spp_avg, tr.h_plays_act+tr.a_plays_act)[0,1]**2:.3f})")
print(f"  -> even perfect knowledge of realized plays explains {np.corrcoef(gp, te.total_err_mkt)[0,1]**2*100:.1f}% of market residual variance; prior pace predicts {np.corrcoef(te.spp_avg, gp)[0,1]**2*100:.1f}% of plays -> upper bound on pace's residual R^2 ~ {np.corrcoef(gp, te.total_err_mkt)[0,1]**2*np.corrcoef(te.spp_avg, gp)[0,1]**2*100:.2f}%")

# ---------- (7) dispersion cross-check with the totals expert's independent sec_per_play ----------
print("\n== (7) 'pace dispersion fell': cross-check with the totals expert's independent sec_per_play (possession clock / plays) ==")
p = pd.read_csv(HERE.parents[0] / "totals" / "pace_team_games.csv")
p = p.sort_values(["team", "season", "gid"])
p["era"] = np.where(p.season <= 2019, 1, 2)
p["spp_r8"] = p.groupby(["team", "era"]).sec_per_play.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
ts = p.groupby(["team", "season"]).sec_per_play.mean().reset_index()
for era, lab in [((2009, 2019), "2009-19"), ((2016, 2019), "2016-19"), ((2023, 2025), "2023-25")]:
    x = p[(p.season >= era[0]) & (p.season <= era[1])].dropna(subset=["spp_r8"]); y = ts[(ts.season >= era[0]) & (ts.season <= era[1])]
    sd_within = y.groupby("season").sec_per_play.std().mean()
    print(f"  {lab}: SD of prior-8 sec/play (totals-expert def) = {x.spp_r8.std():.2f}; SD of team-season means within season = {sd_within:.2f}; corr(prior-8, this game) = {np.corrcoef(x.spp_r8, x.sec_per_play)[0,1]:+.3f}; n={len(x)}")
tg2 = tgf[tgf.game_type == "REG"].dropna(subset=["spp_neut_r8"])
for era, lab in [((2009, 2019), "2009-19"), ((2016, 2019), "2016-19"), ((2023, 2025), "2023-25")]:
    x = tg2[(tg2.season >= era[0]) & (tg2.season <= era[1])].dropna(subset=["spp_neut"])
    sd_within = x.groupby(["season", "team"]).spp_neut.mean().groupby("season").std().mean()
    print(f"  {lab}: SD of prior-8 NEUTRAL sec/play (pace-expert def) = {x.spp_neut_r8.std():.2f}; team-season SD within season = {sd_within:.2f}; corr(prior-8, this game) = {np.corrcoef(x.spp_neut_r8, x.spp_neut)[0,1]:+.3f}")
