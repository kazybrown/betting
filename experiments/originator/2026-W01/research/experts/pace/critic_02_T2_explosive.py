"""critic_02 (T2 explosive offense vs leaky defense -> team totals). Re-derives the expert's numbers
from the expert's feature table (long format, cluster-by-game SEs), then attacks: (1) the multiple-
comparison load behind the one 'significant' test-era coefficient; (2) the distribution of single-
season coefficients (is +25 in 2023-25 unusual vs 2009-19 season-by-season?); (3) a genuine modern-
era OOS test of the 'explosive offenses beat their implied total' watch item (fit 2023 -> test 2024,
fit 2023-24 -> test 2025); (4) influence: leave-one-team-season-out on the 2023-25 coefficient and a
Huber fit; (5) the leak-free season-to-date league reference; (6) the defense-allowed market-free term.
Fit 2009-2019, test 2023-2025 REG as the expert.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from critic_common import *  # noqa
pd.set_option("display.width", 220)

m = load_games_table()
m = add_std_centered(m)
m = m[(m.game_type == "REG") & m.mkt_total.notna() & m.mkt_spread.notna()].copy()
TF = ["expl_off", "expl_def", "epa_off", "epa_def", "pts", "pts_allowed"]
TB = ["team_elo", "opp_elo", "team_pf", "opp_pa", "team_qb", "opp_qb", "is_home", "dome", "wind_f", "div"]


def long(m, v):
    rows = []
    for side, opp in (("h", "a"), ("a", "h")):
        x = pd.DataFrame({"gid": m.gid, "season": m.season, "week": m.week, "train": m.train, "test": m.test,
                          "team": m.home if side == "h" else m.away,
                          "team_pts": m.home_score if side == "h" else m.away_score,
                          "implied_tt": m.implied_home_tt if side == "h" else m.implied_away_tt,
                          "is_home": 1 if side == "h" else 0, "lg_half": m.lg_blend / 2,
                          "team_elo": m.home_pts_vs_avg if side == "h" else m.away_pts_vs_avg,
                          "opp_elo": m.away_pts_vs_avg if side == "h" else m.home_pts_vs_avg,
                          "team_pf": m.h_pf if side == "h" else m.a_pf, "opp_pa": m.a_pa if side == "h" else m.h_pa,
                          "team_qb": (m.home_538_qb_adj if side == "h" else m.away_538_qb_adj).fillna(0) / 25,
                          "opp_qb": (m.away_538_qb_adj if side == "h" else m.home_538_qb_adj).fillna(0) / 25,
                          "dome": m.dome, "wind_f": m.wind_f, "div": m["div"]})
        for f in TF:
            x["off_" + f] = m[f"{side}_{f}_{v}"]; x["opp_" + f] = m[f"{opp}_{f}_{v}"]
        rows.append(x)
    d = pd.concat(rows, ignore_index=True)
    d["y_mkt"] = d.team_pts - d.implied_tt
    d["mismatch"] = d.off_expl_off * d.opp_expl_def * 100
    return d


d = long(m, "r8d"); d = d[d[TB + ["off_expl_off", "opp_expl_def", "lg_half", "implied_tt"]].notna().all(axis=1)].copy()
ds = long(m, "r8s"); ds = ds.loc[d.index]
tr, te = d[d.train], d[d.test]
print(f"sample team-games: train n={len(tr)} test n={len(te)} (expert 4884 / 1504); sanity corr(implied_tt, team_pts) test={np.corrcoef(te.implied_tt, te.team_pts)[0,1]:+.3f}")

print("\n== (0) reproduce ==")
for lab, cols in [("off_expl_off", ["off_expl_off"]), ("opp_expl_def", ["opp_expl_def"]), ("mismatch", ["mismatch"])]:
    f1 = ols(tr.y_mkt, tr[cols], groups=tr.gid); f2 = ols(te.y_mkt, te[cols], groups=te.gid)
    print(f"  market residual ~ {lab:13s} train {f1.params[cols[0]]:+8.3f} (se {f1.bse[cols[0]]:.3f}) p={f1.pvalues[cols[0]]:.3f} | test {f2.params[cols[0]]:+8.3f} (se {f2.bse[cols[0]]:.3f}) p={f2.pvalues[cols[0]]:.3f}")
sd_o = tr.off_expl_off.std()
print(f"  (expert: off_expl test +25.054 se 11.422 p=0.028; 1 SD of off_expl = {sd_o:.4f} -> +25 per unit = {25*sd_o:+.2f} pts per SD)")

print("\n== (1) multiple comparisons: the expert reported 10 specs x 2 feature versions x 2 eras = 40 market-residual coefficient tests in T2 (plus 2 cell tests) ==")
print("  expected number of p<0.05 under the null: 2.0; observed in the expert's log: 2 (off_expl test r8 p=0.028, 'both additive' test r8 p=0.025 -- the same feature) -> exactly the null expectation")

print("\n== (2) single-season coefficients of (team_pts - implied_tt) ~ off_expl (r8d), cluster SE ==")
rows = []
for y, x in d.groupby("season"):
    f = ols(x.y_mkt, x[["off_expl_off"]], groups=x.gid)
    rows.append((y, len(x), f.params["off_expl_off"], f.bse["off_expl_off"]))
r = pd.DataFrame(rows, columns=["season", "n", "coef", "se"])
print("  " + "  ".join(f"{int(y)}:{c:+.0f}({s:.0f})" for y, c, s in zip(r.season, r.coef, r.se)))
old = r[r.season <= 2019]
print(f"  2009-19 season coefs: mean {old.coef.mean():+.1f}, SD {old.coef.std():.1f}, range [{old.coef.min():+.0f},{old.coef.max():+.0f}]; 2023-25 individual seasons: " + ", ".join(f"{int(y)} {c:+.0f}" for y, c in zip(r[r.season>=2023].season, r[r.season>=2023].coef)))
print(f"  -> the 2023-25 pooled +25 is driven by which seasons? (a +25 value would be exceeded by a single 2009-19 season with prob ~{np.mean(old.coef >= 25):.2f})")

print("\n== (3) modern-era OOS test of the watch item: fit market residual ~ off_expl on earlier modern seasons, test the next ==")
for fit_s, test_s in [([2023], 2024), ([2023, 2024], 2025), ([2024], 2025)]:
    a = d[d.season.isin(fit_s)]; b = d[d.season == test_s]
    f = ols(a.y_mkt, a[["off_expl_off"]], groups=a.gid)
    adj = f.params["const"] + f.params["off_expl_off"] * b.off_expl_off
    dm, lo, hi, n = paired_mae_ci(b.team_pts - (b.implied_tt + adj), b.y_mkt)
    w, l, pu = ou_rate(b.implied_tt + adj, b.implied_tt, b.team_pts)
    fb = ols(b.y_mkt, b[["off_expl_off"]], groups=b.gid)
    print(f"  fit {fit_s} (coef {f.params['off_expl_off']:+.1f}) -> test {test_s}: in-sample coef {fb.params['off_expl_off']:+.1f} (p={fb.pvalues['off_expl_off']:.2f}); OOS dMAE {ci_str(dm, lo, hi)} n={n}; team-total O/U vs implied {w}-{l}-{pu} ({w/(w+l):.3f})")
# same with 2009-19 + modern seasons in the fit
for fit_max, test_s in [(2023, 2024), (2024, 2025)]:
    a = d[(d.season <= 2019) | (d.season <= fit_max)]; b = d[d.season == test_s]
    f = ols(a.y_mkt, a[["off_expl_off"]], groups=a.gid)
    adj = f.params["const"] + f.params["off_expl_off"] * b.off_expl_off
    dm, lo, hi, n = paired_mae_ci(b.team_pts - (b.implied_tt + adj), b.y_mkt)
    print(f"  fit 2009-19 + 2023..{fit_max} (coef {f.params['off_expl_off']:+.1f}) -> test {test_s}: OOS dMAE {ci_str(dm, lo, hi)}")

print("\n== (4) influence and robustness of the 2023-25 coefficient ==")
h = sm.RLM(te.y_mkt.values, sm.add_constant(te[["off_expl_off"]].astype(float)), M=sm.robust.norms.HuberT()).fit()
print(f"  Huber fit 2023-25: {h.params['off_expl_off']:+.1f} (se {h.bse['off_expl_off']:.1f})")
te2 = te.copy(); te2["ts"] = te2.team + "_" + te2.season.astype(str)
full = ols(te.y_mkt, te[["off_expl_off"]], groups=te.gid).params["off_expl_off"]
infl = []
for ts_, idx in te2.groupby("ts").indices.items():
    x = te2.drop(te2.index[idx]); f = ols(x.y_mkt, x[["off_expl_off"]], groups=x.gid)
    infl.append((ts_, len(idx), f.params["off_expl_off"] - full, te2.iloc[idx].y_mkt.mean(), te2.iloc[idx].off_expl_off.mean()))
infl = pd.DataFrame(infl, columns=["team_season", "n", "d_coef_when_dropped", "mean_resid", "mean_off_expl"]).sort_values("d_coef_when_dropped")
print("  team-seasons whose removal most reduces the coefficient (top 6):")
print(infl.head(6).round(3).to_string(index=False))
print(f"  dropping the top 3 together: coef = {ols(te2[~te2.ts.isin(infl.team_season.head(3))].y_mkt, te2[~te2.ts.isin(infl.team_season.head(3))][['off_expl_off']], groups=te2[~te2.ts.isin(infl.team_season.head(3))].gid).params['off_expl_off']:+.1f}")

print("\n== (5) leak-free season-to-date league reference (r8s) instead of the prior-season reference ==")
trs, tes = ds[ds.train], ds[ds.test]
for lab, x, xs in [("train", tr, trs), ("test", te, tes)]:
    f = ols(xs.y_mkt, xs[["off_expl_off"]], groups=xs.gid)
    print(f"  {lab}: market residual ~ off_expl (std ref) {f.params['off_expl_off']:+.2f} (se {f.bse['off_expl_off']:.2f}) p={f.pvalues['off_expl_off']:.3f}  | corr(expert ref, std ref)={np.corrcoef(x.off_expl_off, xs.off_expl_off)[0,1]:.3f}")

print("\n== (6) defense-allowed explosive rate, market-free team-points baseline: calibration + std ref + rolling-origin ==")
fb_, pb_ = fit_pred(tr, te, TB, y="team_pts", offset="lg_half", groups="gid"); resb = te.team_pts.values - pb_
for lab, x_tr, x_te, c in [("expert ref", tr, te, "opp_expl_def"), ("std ref", trs, tes, "opp_expl_def")]:
    f, p = fit_pred(x_tr, x_te, TB + [c], y="team_pts", offset="lg_half", groups="gid")
    adj = f.params[c] * (x_te[c] - x_tr[c].mean()); cal = ols(resb, pd.DataFrame({"adj": adj}))
    dm, lo, hi, n = paired_mae_ci(x_te.team_pts - p, x_te.team_pts - pb_)
    print(f"  {lab:11s} b={f.params[c]:+.1f} (p={f.pvalues[c]:.3f}) | calib slope {cal.params['adj']:+.2f} (se {cal.bse['adj']:.2f}) | dMAE {ci_str(dm, lo, hi)}")
acc = []
for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
    a = d[(d.season < Y)]; b = d[d.season == Y]
    f0, p0 = fit_pred(a, b, TB, y="team_pts", offset="lg_half", groups="gid"); f1, p1 = fit_pred(a, b, TB + ["opp_expl_def", "off_expl_off"], y="team_pts", offset="lg_half", groups="gid")
    acc.append((Y, mae(p1, b.team_pts) - mae(p0, b.team_pts)))
print("  rolling-origin BASE+expl(off,def) minus BASE: " + "  ".join(f"{y}:{v:+.3f}" for y, v in acc) + f" | mean {np.mean([v for _, v in acc]):+.3f}")
print(f"  market prices it: corr(opp_expl_def prior, implied_tt) train {np.corrcoef(tr.opp_expl_def, tr.implied_tt)[0,1]:+.3f}; residual coef vs market train {ols(tr.y_mkt, tr[['opp_expl_def']], groups=tr.gid).params['opp_expl_def']:+.1f} (p={ols(tr.y_mkt, tr[['opp_expl_def']], groups=tr.gid).pvalues['opp_expl_def']:.2f})")
