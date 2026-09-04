"""05: RECOMMENDATION pass (Theory 4). Everything OOS = fit 2009-2019, test 2023-2025 REG.
(1) Modern-era in-sample check: are the pace / style / efficiency effects present in 2023-25 at all?
(2) Calibration slopes: for each candidate adjustment (train coefficient x prior feature), regress the
    OOS residual of the market-free BASE on the adjustment. slope ~1 = right-sized, ~0 = no OOS value.
    Plus MAE and RMSE deltas with paired bootstrap CIs.
(3) Head-to-head of concrete pace rules on the BASE total: none | spec bucket rule (+/-1..2, implemented
    as FF +1.5, SS -1.5, F+M +0.75, S+M -0.75) | linear -0.5 x spp_avg capped +/-1 | linear from train fit.
(4) Efficiency alternative: EPA/play (for+against, both teams) and points per drive.
(5) Team-total: explosive-plays-allowed adjustment on the market-free team-points baseline.
(6) Weekly feature spec.
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experts" / "totals"))
from common import mae, paired_mae_ci, ou_rate
pd.set_option("display.width", 220)

m = pd.read_csv(HERE / "_game_features.csv", low_memory=False)
m["dome"] = m.is_dome.astype(int)
BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "div"]
V = "r8d"


def combos(d, v):
    d = d.copy()
    d["spp_avg"] = (d[f"h_spp_neut_{v}"] + d[f"a_spp_neut_{v}"]) / 2
    d["sppr_avg"] = (d[f"h_spp_neut_run_{v}"] + d[f"a_spp_neut_run_{v}"]) / 2
    d["gplays_avg"] = (d[f"h_game_plays_{v}"] + d[f"a_game_plays_{v}"]) / 2
    d["proe_sum"] = d[f"h_proe_{v}"] + d[f"a_proe_{v}"]
    d["pr_sum"] = d[f"h_pr_neut_{v}"] + d[f"a_pr_neut_{v}"]
    d["ppd_sum"] = d[f"h_ppd_{v}"] + d[f"a_ppd_{v}"]
    d["epa_sum"] = d[f"h_epa_off_{v}"] + d[f"a_epa_off_{v}"] + d[f"h_epa_def_{v}"] + d[f"a_epa_def_{v}"]
    d["expl_def_sum"] = d[f"h_expl_def_{v}"] + d[f"a_expl_def_{v}"]
    d["expl_off_sum"] = d[f"h_expl_off_{v}"] + d[f"a_expl_off_{v}"]
    return d


def ols(y, X):
    X = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(cov_type="HC1")


def fit_pred(tr, te, cols, y="total_pts", offset="lg_blend"):
    f = ols(tr[y] - tr[offset], tr[cols])
    return f, np.asarray(te[offset] + f.predict(sm.add_constant(te[cols].astype(float), has_constant="add")))


def rmse(p, a):
    return float(np.sqrt(np.nanmean((np.asarray(p, float) - np.asarray(a, float)) ** 2)))


def boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


CANDS = ["spp_avg", "sppr_avg", "gplays_avg", "proe_sum", "pr_sum", "ppd_sum", "epa_sum", "expl_def_sum", "expl_off_sum"]
d = combos(m, V)
d = d[(d.game_type == "REG") & d.mkt_total.notna() & d[BASE + ["lg_blend"] + CANDS].notna().all(axis=1)].copy()
tr, te = d[d.train], d[d.test]
print(f"sample (prior-8-game league-relative features): train 2009-2019 n={len(tr)} | test 2023-2025 n={len(te)}")

print("\n== (1) modern-era IN-SAMPLE check: BASE + x fit on 2023-2025 only (n=%d) vs the 2009-2019 fit ==" % len(te))
print(f"  {'x':13s} {'2009-19 coef (se) p':>24s}   {'2023-25 coef (se) p':>24s}   corr(x, total) 09-19 | 23-25")
for c in CANDS:
    f1 = ols(tr.total_pts - tr.lg_blend, tr[BASE + [c]]); f2 = ols(te.total_pts - te.lg_blend, te[BASE + [c]])
    print(f"  {c:13s} {f1.params[c]:+9.3f} ({f1.bse[c]:.3f}) p={f1.pvalues[c]:.3f}   {f2.params[c]:+9.3f} ({f2.bse[c]:.3f}) p={f2.pvalues[c]:.3f}   "
          f"{stats.pearsonr(tr[c], tr.total_pts)[0]:+.3f} | {stats.pearsonr(te[c], te.total_pts)[0]:+.3f}")

print("\n== (2) OOS calibration of train-fit adjustments on top of BASE (test 2023-25) ==")
fb, pb = fit_pred(tr, te, BASE)
res_b = te.total_pts.values - pb
print(f"  BASE: MAE={mae(pb, te.total_pts):.3f} RMSE={rmse(pb, te.total_pts):.3f} | market MAE={mae(te.mkt_total, te.total_pts):.3f} RMSE={rmse(te.mkt_total, te.total_pts):.3f}")
print(f"  {'adj = b*x':13s} {'b (train)':>10s} {'mean|adj|':>9s} {'calib slope (se) p':>24s}   dMAE [95% CI]            dRMSE   in-sample train dMAE")
for c in CANDS:
    f, p = fit_pred(tr, te, BASE + [c])
    adj = f.params[c] * (te[c] - tr[c].mean())         # centered adjustment
    cal = ols(res_b, pd.DataFrame({"adj": adj}))
    dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
    _, ptr = fit_pred(tr, tr, BASE + [c]); _, pbt = fit_pred(tr, tr, BASE)
    print(f"  {c:13s} {f.params[c]:+10.3f} {np.abs(adj).mean():9.2f} {cal.params['adj']:+8.2f} ({cal.bse['adj']:.2f}) p={cal.pvalues['adj']:.3f}   "
          f"{dm:+.3f} [{lo:+.3f},{hi:+.3f}]   {rmse(p, te.total_pts)-rmse(pb, te.total_pts):+.3f}   {mae(ptr, tr.total_pts)-mae(pbt, tr.total_pts):+.3f}")

print("\n== (3) concrete PACE rules applied to the BASE total (OOS 2023-25) ==")
lo_q, hi_q = np.percentile(pd.concat([tr[f"h_spp_neut_{V}"], tr[f"a_spp_neut_{V}"]]), [25, 75])
def cls(x): return np.where(x <= lo_q, "F", np.where(x >= hi_q, "S", "M"))
for frame in (tr, te, d):
    frame["hc"] = cls(frame[f"h_spp_neut_{V}"]); frame["ac"] = cls(frame[f"a_spp_neut_{V}"])
print(f"  quartile cutoffs (train): fast <= {lo_q:+.2f} s/play vs league, slow >= {hi_q:+.2f}; mean spp_avg by bucket (test): "
      + ", ".join(f"{b}={te[(te.hc+te.ac).replace({'SF':'FS'}).eq(b)].spp_avg.mean():+.2f}" for b in ["FF", "FM", "MM", "FS", "SM", "SS"] if ((te.hc+te.ac).replace({'SF':'FS'}) == b).any()))
def spec_rule(fr):
    b = fr.hc + fr.ac
    return np.select([b == "FF", b == "SS", b.isin(["FM", "MF"]), b.isin(["SM", "MS"])], [1.5, -1.5, 0.75, -0.75], 0.0)
f_spp, _ = fit_pred(tr, te, BASE + ["spp_avg"])
rules = {"no pace adj (BASE)": np.zeros(len(te)),
         "spec bucket rule (+/-1.5 FF/SS, +/-0.75 mixed)": spec_rule(te),
         "spec rule x0.5 (+/-0.75 FF/SS, +/-0.375 mixed)": 0.5 * spec_rule(te),
         "linear -0.5*spp_avg, cap +/-1.0": np.clip(-0.5 * te.spp_avg.values, -1, 1),
         "linear train-fit b*spp_avg (b=%.2f), no cap" % f_spp.params["spp_avg"]: f_spp.params["spp_avg"] * (te.spp_avg.values - tr.spp_avg.mean()),
         "linear -0.25*spp_avg, cap +/-0.5": np.clip(-0.25 * te.spp_avg.values, -0.5, 0.5)}
print(f"  {'rule':52s} {'MAE':>7s} {'RMSE':>7s}  dMAE vs BASE [95% CI]     dRMSE   mean|adj|  O/U vs mkt")
for lab, adj in rules.items():
    p = pb + adj
    dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
    w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
    print(f"  {lab:52s} {mae(p, te.total_pts):7.3f} {rmse(p, te.total_pts):7.3f}  {dm:+.3f} [{lo:+.3f},{hi:+.3f}]   {rmse(p, te.total_pts)-rmse(pb, te.total_pts):+.3f}   {np.abs(adj).mean():6.2f}   {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
# same rules in the pre-2020 rolling-origin era (extra OOS evidence, labelled)
print("  rolling-origin 2013-2019 (fit < Y): mean dMAE vs BASE of each rule")
acc = {k: [] for k in rules}
for Y in range(2013, 2020):
    trY, teY = d[d.season < Y], d[d.season == Y]
    fY, pY = fit_pred(trY, teY, BASE); fs, _ = fit_pred(trY, teY, BASE + ["spp_avg"])
    rY = {"no pace adj (BASE)": np.zeros(len(teY)), "spec bucket rule (+/-1.5 FF/SS, +/-0.75 mixed)": spec_rule(teY),
          "spec rule x0.5 (+/-0.75 FF/SS, +/-0.375 mixed)": 0.5 * spec_rule(teY),
          "linear -0.5*spp_avg, cap +/-1.0": np.clip(-0.5 * teY.spp_avg.values, -1, 1),
          list(rules)[4]: fs.params["spp_avg"] * (teY.spp_avg.values - trY.spp_avg.mean()),
          "linear -0.25*spp_avg, cap +/-0.5": np.clip(-0.25 * teY.spp_avg.values, -0.5, 0.5)}
    for k in rules:
        acc[k].append(mae(pY + rY[k], teY.total_pts) - mae(pY, teY.total_pts))
for k in rules:
    print(f"    {k:52s} mean dMAE={np.mean(acc[k]):+.3f}  seasons better: {sum(a < 0 for a in acc[k])}/7")

print("\n== (4) efficiency alternative on the BASE total (OOS 2023-25): EPA/play (for+against, both teams) and points per drive ==")
f_epa, _ = fit_pred(tr, te, BASE + ["epa_sum"]); f_ppd, _ = fit_pred(tr, te, BASE + ["ppd_sum"])
for lab, adj in {"EPA: train-fit b*epa_sum (b=%.2f)" % f_epa.params["epa_sum"]: f_epa.params["epa_sum"] * (te.epa_sum.values - tr.epa_sum.mean()),
                 "EPA: 4.0*epa_sum, cap +/-3": np.clip(4.0 * te.epa_sum.values, -3, 3),
                 "EPA: 3.0*epa_sum, cap +/-2": np.clip(3.0 * te.epa_sum.values, -2, 2),
                 "PPD: train-fit b*ppd_sum (b=%.2f)" % f_ppd.params["ppd_sum"]: f_ppd.params["ppd_sum"] * (te.ppd_sum.values - tr.ppd_sum.mean()),
                 "EPA 4.0 + pace -0.5*spp (both capped)": np.clip(4.0 * te.epa_sum.values, -3, 3) + np.clip(-0.5 * te.spp_avg.values, -1, 1)}.items():
    p = pb + adj
    dm, lo, hi, n = paired_mae_ci(te.total_pts - p, te.total_pts - pb)
    dmm, lom, him, _ = paired_mae_ci(te.total_pts - p, te.total_err_mkt)
    w, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
    print(f"  {lab:42s} MAE={mae(p, te.total_pts):.3f} dMAE vs BASE {dm:+.3f} [{lo:+.3f},{hi:+.3f}]  vs market {dmm:+.3f} [{lom:+.3f},{him:+.3f}]  dRMSE {rmse(p, te.total_pts)-rmse(pb, te.total_pts):+.3f}  mean|adj|={np.abs(adj).mean():.2f}  O/U {w}-{l}-{pu} ({w/max(w+l,1):.3f})")
acc = []
for Y in range(2013, 2020):
    trY, teY = d[d.season < Y], d[d.season == Y]
    fY, pY = fit_pred(trY, teY, BASE)
    acc.append(mae(pY + np.clip(4.0 * teY.epa_sum.values, -3, 3), teY.total_pts) - mae(pY, teY.total_pts))
print(f"  rolling-origin 2013-2019: EPA 4.0*epa_sum cap 3 -> mean dMAE vs BASE {np.mean(acc):+.3f}, seasons better {sum(a < 0 for a in acc)}/7")
print(f"  epa_sum scale: train SD={tr.epa_sum.std():.3f}, so 4.0*epa_sum has SD {4*tr.epa_sum.std():.2f} pts; 1st/99th pct of adj in test: "
      f"{np.percentile(4*te.epa_sum, 1):+.2f} / {np.percentile(4*te.epa_sum, 99):+.2f}")

print("\n== (5) TEAM TOTAL: explosive-plays-allowed by the opponent's defense, on the market-free team-points baseline (OOS 2023-25) ==")
rows = []
for side, opp in (("h", "a"), ("a", "h")):
    x = pd.DataFrame({"gid": d.gid, "season": d.season, "train": d.train, "test": d.test, "team_pts": d.home_score if side == "h" else d.away_score,
                      "implied_tt": d.implied_home_tt if side == "h" else d.implied_away_tt, "lg_half": d.lg_blend / 2, "is_home": 1 if side == "h" else 0,
                      "team_elo": d.home_pts_vs_avg if side == "h" else d.away_pts_vs_avg, "opp_elo": d.away_pts_vs_avg if side == "h" else d.home_pts_vs_avg,
                      "team_pf": d.h_pf if side == "h" else d.a_pf, "opp_pa": d.a_pa if side == "h" else d.h_pa,
                      "team_qb": (d.home_538_qb_adj if side == "h" else d.away_538_qb_adj).fillna(0) / 25, "opp_qb": (d.away_538_qb_adj if side == "h" else d.home_538_qb_adj).fillna(0) / 25,
                      "dome": d.dome, "wind_f": d.wind_f, "div": d["div"], "opp_expl_def": d[f"{opp}_expl_def_{V}"], "off_expl": d[f"{side}_expl_off_{V}"],
                      "off_epa": d[f"{side}_epa_off_{V}"], "opp_def_epa": d[f"{opp}_epa_def_{V}"]})
    rows.append(x)
L = pd.concat(rows, ignore_index=True).dropna(subset=["implied_tt"]); Ltr, Lte = L[L.train], L[L.test]
TB = ["team_elo", "opp_elo", "team_pf", "opp_pa", "team_qb", "opp_qb", "is_home", "dome", "wind_f", "div"]
fb2, pb2 = fit_pred(Ltr, Lte, TB, y="team_pts", offset="lg_half"); resb = Lte.team_pts.values - pb2
print(f"  BASE team pts: MAE={mae(pb2, Lte.team_pts):.3f} | implied_tt MAE={mae(Lte.implied_tt, Lte.team_pts):.3f}")
for c in ["opp_expl_def", "off_expl", "off_epa", "opp_def_epa"]:
    f, p = fit_pred(Ltr, Lte, TB + [c], y="team_pts", offset="lg_half")
    adj = f.params[c] * (Lte[c] - Ltr[c].mean()); cal = ols(resb, pd.DataFrame({"adj": adj}))
    dm, lo, hi, n = paired_mae_ci(Lte.team_pts - p, Lte.team_pts - pb2)
    print(f"  + {c:13s} b={f.params[c]:+8.3f} (p={f.pvalues[c]:.3f}) per SD {f.params[c]*Ltr[c].std():+.2f} | calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f}) | dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}] mean|adj|={np.abs(adj).mean():.2f}")
adj = np.clip(30.0 * Lte.opp_expl_def.values, -1.5, 1.5); p = pb2 + adj
dm, lo, hi, n = paired_mae_ci(Lte.team_pts - p, Lte.team_pts - pb2)
print(f"  rule: team_total += 30 * (opp explosive-allowed rate - league), cap +/-1.5 -> dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}], mean|adj|={np.abs(adj).mean():.2f}")

print("\n== (6) WEEKLY FEATURE SPEC (what to compute from pbp each week; all prior-only, league-relative) ==")
spec = pd.DataFrame([
    ["spp_neut", "game-clock seconds between consecutive offensive snaps of the same drive, Q1-Q3, |score diff|<=8, excl. last 2:00 of H1, 0<dt<=60", "prior 8 games, weighted by gap count", "pace"],
    ["plays / game_plays", "offensive pass+run plays; game_plays = own + opponent plays in the team's games", "prior 8 games", "pace (diagnostic only)"],
    ["pr_neut / proe", "pass share of neutral pass+run plays; PROE = pass - xpass (logit on down/dist/field pos/score x time/clock/wp), x100", "prior 8 games", "style (market prices it; no adj)"],
    ["epa_off / epa_def", "EPA per pass+run play for / against", "prior 8 games", "efficiency (recommended total adj)"],
    ["ppd", "offensive drive points / offensive drives (XP included)", "prior 8 games", "efficiency (weaker than EPA)"],
    ["expl_off / expl_def", "(pass>=20 yds or rush>=10 yds) per play, for / allowed", "prior 8 games", "team-total (defense-allowed only; small)"],
], columns=["feature", "definition", "window", "use"])
print(spec.to_string(index=False))
print("  league-relative = subtract the prior-season league mean (2019 league mean when the prior season has no pbp).")
print("  minimum 4 prior games; before that fall back to prior-season team mean, else league mean (the K=4 blend behaved the same as rolling-8 in every test).")
