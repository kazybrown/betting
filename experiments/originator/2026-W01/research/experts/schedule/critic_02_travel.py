"""CRITIC 02 - T4a (west team @ 1pm ET), T4b (3-zone traveller), T4c (primetime west edge).
(a) T4a: proper 95% CI for the west team's disadvantage; power to detect the spec's -0.6; 1999-2008 market-only check.
(b) T4b: multiple-comparisons count over the travel buckets the expert screened; season-clustered inference; dose-response (0/1/2/3 zones);
    kickoff-window split; exclusions (relocated LA/LAC/LV hosts, SEA/SF travellers); distance-based alternative definition;
    rolling-origin OOS of '+k to the traveller' with k fit on prior seasons only.
(c) T4c: primetime west edge split by west team home/away; same-zone primetime placebo; 'away team in primetime' confound; rolling-origin."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule"); sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from common import build, desc, ols, ats_side, tz_offset
from kit import load_games
pd.set_option("display.width", 250)
rng = np.random.default_rng(7)
m = build(); m = m[~m.neutral].copy()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["xc"] = (d.tz_diff.abs() == 3).astype(float); d["z"] = d.tz_diff.abs()
d["west_early"] = (d.away_off <= -2) & d.early
d["win"] = np.select([d.early, d.late, d.primetime], ["1pm", "4pm", "prime"], default="other")

print("=== (a) T4a: west (MT/PT) team at 1pm ET. Theory: west team -0.4..-0.8 => home residual +0.4..+0.8 ===")
for lab, x in [("PT only ALL", d[(d.away_off == -3) & d.early]), ("MT/PT ALL", d[d.west_early]), ("MT/PT FIT", d[d.west_early & d.fit]), ("MT/PT TEST", d[d.west_early & d.test])]:
    for c in ["err_mkt", "err_rate", "err_nraw"]:
        n, mu, se, p = desc(x[c]); lo, hi = mu - 1.96 * se, mu + 1.96 * se
        print("  %-12s %-8s n=%3d home resid %+.2f 95%% CI [%+.2f, %+.2f]  -> spec's +0.6 %s" % (lab, c, n, mu, lo, hi, "EXCLUDED" if hi < 0.6 else "inside CI"))
# power: with se 0.68, what effect could be detected at 80% power?
print("  minimum detectable effect (80%% power, alpha .05) with se 0.68: %.2f pts" % (2.8 * 0.68))
g = load_games(min_season=1999); g = g[(g.game_type == "REG") & g.mkt_spread.notna() & g.location.eq("Home") & g.season.between(1999, 2008)].dropna(subset=["gametime"]).copy()
g["kick"] = g.gametime.str.slice(0, 2).astype(int) + g.gametime.str.slice(3, 5).astype(int) / 60
g["aoff"] = [tz_offset(t, dd) for t, dd in zip(g.away_team, g.gameday)]
x = g[(g.aoff <= -2) & g.kick.between(12.5, 13.5)]
n, mu, se, p = desc(x.spread_err_mkt); w, l, ps, pct, pv = ats_side(x.spread_err_mkt, 1)
print("  1999-2008 market-only: MT/PT away @ 1pm ET n=%d home vs close %+.2f (se %.2f p=%.2f); fade-west ATS %d-%d %.3f" % (n, mu, se, p, w, l, pct))

print("\n=== (b) T4b: 3-zone traveller ===")
print("  (b1) multiple comparisons: the buckets screened in 04_travel.py section A (ALL sample, err_mkt): p-values")
buckets = {"PT@1pm": (d.away_off == -3) & d.early, "MT@1pm": (d.away_off == -2) & d.early, "ET@4pm out west": (d.away_off == 0) & (d.home_off <= -2) & d.late,
           "ET prime out west": (d.away_off == 0) & (d.home_off <= -2) & d.primetime, "MT/PT prime in east": (d.away_off <= -2) & (d.home_off == 0) & d.primetime,
           "xc east": d.tz_diff == 3, "xc west": d.tz_diff == -3, "2 zones": d.z == 2, "1 zone": d.z == 1}
ps = {}
for k, mask in buckets.items():
    n, mu, se, p = desc(d[mask].err_mkt); ps[k] = p; print("    %-20s n=%4d err_mkt %+.2f (se %.2f) p=%.3f" % (k, n, mu, se, p))
print("    Bonferroni over %d buckets: xc east p*%d = %.2f; pooled xc (FE, p=0.003 per expert) * %d = %.3f" % (len(ps), len(ps), ps["xc east"] * len(ps), len(ps), 0.003 * len(ps)))
print("  (b2) season-clustered inference (season means as the unit):")
for lo, hi in [(2009, 2017), (2018, 2025), (2022, 2025), (2009, 2025)]:
    x = d[(d.xc == 1) & d.season.between(lo, hi)]; sm_ = x.groupby("season").err_mkt.mean()
    r = smf.ols("err_mkt ~ xc", data=d[d.season.between(lo, hi)]).fit(cov_type="cluster", cov_kwds={"groups": d[d.season.between(lo, hi)].season})
    print("    %d-%d: %d seasons, %d negative; mean of season means %+.2f (se %.2f, t-test p=%.3f) | cluster-robust xc coef %+.2f (se %.2f p=%.3f)" %
          (lo, hi, len(sm_), (sm_ < 0).sum(), sm_.mean(), sm_.std() / np.sqrt(len(sm_)), stats.ttest_1samp(sm_, 0).pvalue, r.params["xc"], r.bse["xc"], r.pvalues["xc"]))
print("  (b3) dose-response: home residual vs market by |zones| (ALL and 2018-25)")
for lab, x in [("ALL", d), ("2018-25", d[d.season >= 2018]), ("TEST", d[d.test])]:
    cells = []
    for z in [0, 1, 2, 3]:
        n, mu, se, p = desc(x[x.z == z].err_mkt); cells.append("z=%d n=%4d %+.2f (se %.2f)" % (z, n, mu, se))
    print("    %-8s " % lab + " | ".join(cells))
print("  (b4) 3-zone games by kickoff window (home residual vs market / vs ratings; traveller ATS)")
for lab, x in [("ALL", d[d.xc == 1]), ("2018-25", d[(d.xc == 1) & (d.season >= 2018)])]:
    for wn in ["1pm", "4pm", "prime"]:
        y = x[x.win == wn]; n, mu, se, p = desc(y.err_mkt); w, l, ps_, pct, pv = ats_side(y.err_mkt, -1)
        print("    %-8s %-6s n=%3d err_mkt %+.2f (se %.2f p=%.2f) err_rate %+.2f | traveller ATS %d-%d %.3f | east-bound share %.2f" % (lab, wn, n, mu, se, p, y.err_rate.mean(), w, l, pct, (y.tz_diff == 3).mean()))
print("  (b5) exclusions (2018-25, err_mkt):")
x = d[(d.xc == 1) & (d.season >= 2018)]
for lab, mask in [("all", x.xc == 1), ("drop LA/LAC/LV hosts", ~x.home_team.isin(["LA", "LAR", "LAC", "LV", "OAK"])), ("drop SEA/SF travellers", ~x.away_team.isin(["SEA", "SF"])),
                  ("drop both", ~x.home_team.isin(["LA", "LAR", "LAC", "LV", "OAK"]) & ~x.away_team.isin(["SEA", "SF"])), ("drop 2020", x.season != 2020), ("east-bound only", x.tz_diff == 3), ("west-bound only", x.tz_diff == -3)]:
    y = x[mask]; n, mu, se, p = desc(y.err_mkt); w, l, ps_, pct, pv = ats_side(y.err_mkt, -1)
    print("    %-24s n=%3d %+.2f (se %.2f p=%.3f) | traveller ATS %d-%d %.3f" % (lab, n, mu, se, p, w, l, pct))
print("  (b6) alternative definition: distance >= 2000 mi (any zones) vs 3 zones; and 3 zones but < 2200 mi")
for lab, mask in [("dist>=2000", d.dist >= 2000), ("dist>=2000 & z<3", (d.dist >= 2000) & (d.z < 3)), ("z=3 & dist<2200", (d.z == 3) & (d.dist < 2200)), ("z=3 & dist>=2200", (d.z == 3) & (d.dist >= 2200)), ("z=2 & dist>=1500", (d.z == 2) & (d.dist >= 1500))]:
    for sub, x in [("ALL", d[mask]), ("2018-25", d[mask & (d.season >= 2018)])]:
        n, mu, se, p = desc(x.err_mkt); print("    %-20s %-8s n=%4d err_mkt %+.2f (se %.2f p=%.3f)" % (lab, sub, n, mu, se, p))
print("  (b7) rolling-origin OOS: k (pts to traveller) fit on all seasons < s (MAE-min on rating-only line), applied in season s; base = ORIGINATOR proxy (c1)")
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["c1"] = 0.46 * (d.nraw_line + d.nfelo_bye_pts + d.nfelo_tz_pts) + 0.54 * d.rate_line - 0.15 * d.rd
rows = []
for s in range(2014, 2026):
    f = d[(d.season < s) & (d.xc == 1)]; t = d[(d.season == s) & (d.xc == 1)]
    ks = np.arange(-1, 4.01, 0.25); k = ks[int(np.argmin([(f.margin + f.rate_line + kk).abs().mean() for kk in ks]))]
    e0 = t.margin + t.c1; e1 = e0 + k
    rows.append((s, k, len(t), e0.abs().mean(), e1.abs().mean(), e0.mean()))
R = pd.DataFrame(rows, columns=["season", "k_fit", "n", "MAE_base", "MAE_adj", "bias_home_base"]); print(R.round(3).to_string(index=False))
tot = R.n.sum(); print("    pooled 2014-25 affected n=%d: MAE %.4f -> %.4f (d=%+.4f); seasons improved %d/%d" % (tot, (R.MAE_base * R.n).sum() / tot, (R.MAE_adj * R.n).sum() / tot, ((R.MAE_adj - R.MAE_base) * R.n).sum() / tot, (R.MAE_adj < R.MAE_base).sum(), len(R)))
# fixed +1.0 rule evaluated OOS from 2018 (the regime the expert identified) - but 2018-21 was used to identify it, so only 2022-25 is clean
for lo in [2018, 2022]:
    t = d[(d.season >= lo) & (d.xc == 1)]; e0 = t.margin + t.c1; e1 = e0 + 1.0; diff = e1.abs().values - e0.abs().values
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    print("    fixed +1.0 rule on xc games %d-25 n=%d: MAE %.3f -> %.3f (d=%+.3f, CI [%+.3f, %+.3f]); bias(home) %+.2f -> %+.2f" % (lo, len(t), e0.abs().mean(), e1.abs().mean(), diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), e0.mean(), e1.mean()))

print("\n=== (c) T4c: primetime west edge ===")
p_ = d[d.primetime].copy(); p_["west_sgn"] = -np.sign(p_.tz_diff)      # +1 = home is the more-western team
p_["west_edge"] = p_.err_mkt * p_.west_sgn
print("  (c1) same-zone primetime placebo: home residual vs market n=%d %+.2f (se %.2f) | away-team edge in ALL primetime (tz any): %+.2f (se %.2f, n=%d)" %
      (*desc(p_[p_.tz_diff == 0].err_mkt)[:3], -p_.err_mkt.mean(), p_.err_mkt.std() / np.sqrt(len(p_)), len(p_)))
print("  (c2) west edge split by whether the western team is HOME or AWAY (tz != 0):")
for lab, x in [("ALL", p_[p_.tz_diff != 0]), ("FIT", p_[(p_.tz_diff != 0) & p_.fit]), ("TEST", p_[(p_.tz_diff != 0) & p_.test])]:
    for side, mask in [("west team HOME", x.west_sgn == 1), ("west team AWAY", x.west_sgn == -1)]:
        y = x[mask]; n, mu, se, p = desc(y.west_edge); w, l, ps_, pct, pv = ats_side(y.err_mkt, y.west_sgn)
        print("    %-4s %-14s n=%3d west edge vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f | west ATS %d-%d %.3f" % (lab, side, n, mu, se, p, (y.err_rate * y.west_sgn).mean(), w, l, pct))
print("  (c3) OLS on primetime games: err_mkt ~ west_sgn + away_dummy-free check: err_mkt ~ tz_diff sign, with home-team & away-team FE (HC1)")
for lab, x in [("ALL", p_[p_.tz_diff != 0]), ("2009-25 all primetime incl same zone", p_)]:
    x = x.copy(); x["ws"] = x.west_sgn
    r = smf.ols("err_mkt ~ ws + C(home_team) + C(away_team)", data=x).fit(cov_type="HC1"); r0 = smf.ols("err_mkt ~ ws", data=x).fit(cov_type="HC1")
    print("    %-38s ws no FE %+.2f (se %.2f p=%.3f) | team FE %+.2f (se %.2f p=%.3f) n=%d" % (lab, r0.params["ws"], r0.bse["ws"], r0.pvalues["ws"], r.params["ws"], r.bse["ws"], r.pvalues["ws"], len(x)))
print("  (c4) by season (west edge vs market, n): ", {int(s): (round(x.west_edge.mean(), 1), len(x)) for s, x in p_[p_.tz_diff != 0].groupby("season")})
sm_ = p_[p_.tz_diff != 0].groupby("season").west_edge.mean(); print("    season means: %d/%d positive, mean %+.2f se %.2f p=%.3f" % ((sm_ > 0).sum(), len(sm_), sm_.mean(), sm_.std() / np.sqrt(len(sm_)), stats.ttest_1samp(sm_, 0).pvalue))
print("  (c5) rolling-origin OOS of '+k to the more-western team in primetime', k fit on prior seasons (MAE-min on err_rate*west_sgn), base c1:")
rows = []
for s in range(2014, 2026):
    f = p_[(p_.season < s) & (p_.tz_diff != 0)]; t = p_[(p_.season == s) & (p_.tz_diff != 0)]
    ks = np.arange(-1, 3.01, 0.25); k = ks[int(np.argmin([((f.margin + f.rate_line) * f.west_sgn - kk).abs().mean() for kk in ks]))]
    e0 = t.margin + t.c1; e1 = e0 - k * t.west_sgn; rows.append((s, k, len(t), e0.abs().mean(), e1.abs().mean()))
R = pd.DataFrame(rows, columns=["season", "k_fit", "n", "MAE_base", "MAE_adj"]); tot = R.n.sum()
print("    " + R.round(3).to_string(index=False).replace("\n", "\n    "))
print("    pooled 2014-25 n=%d MAE %.4f -> %.4f (d=%+.4f); seasons improved %d/%d" % (tot, (R.MAE_base * R.n).sum() / tot, (R.MAE_adj * R.n).sum() / tot, ((R.MAE_adj - R.MAE_base) * R.n).sum() / tot, (R.MAE_adj < R.MAE_base).sum(), len(R)))
print("  (c6) overlap: primetime & 3-zone games n=%d: west edge %+.2f (se %.2f); primetime 1-2 zones n=%d: %+.2f (se %.2f)" %
      (*desc(p_[(p_.z == 3)].west_edge)[:3], *desc(p_[p_.z.isin([1, 2])].west_edge)[:3]))
