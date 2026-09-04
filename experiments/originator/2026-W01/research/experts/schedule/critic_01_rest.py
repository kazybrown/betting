"""CRITIC 01 - T1a/T1b/T2/T3 (short rest, TNF totals, bye, rest differential).
(a) verify hfa_mod = base + tz + div + surf + byes and nfelo_dif_base = elo dif + hfa_mod + qb  -> rebuild the ORIGINATOR proxy AS SPECIFIED in the README
    (PFF/Cole HFA = (hfa_mod + tz)/25), which the expert did NOT do (his 'rating-only' line strips byes/tz, the README build does not).
(b) 1999-2025 market-only sample: asymmetric short-rest counts and TNF totals (larger n than the expert's 2009+ nfelo-joined sample).
(c) nfelo bye mod by season (is the home/away asymmetry a time artifact?).
(d) linear rest slope vs bye dummy: does the continuous term add anything once the bye is in? Robust (LAD/Huber) slopes.
(e) placebo: rest_diff permuted within season -> null distribution of the slope; and 'next-week rest diff' placebo.
(f) rolling-origin bias reduction on affected games with a paired bootstrap CI.
(g) bye-team bias on the correctly-specified ORIGINATOR proxy: as built (nfelo mods via hfa_mod + spec +0.75) vs candidate."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule"); sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import build, desc, ols, ats_side
from kit import load_games
pd.set_option("display.width", 250)
rng = np.random.default_rng(42)
m = build()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
for c in ["hfa_base_mod", "home_time_advantage_mod", "div_game_mod", "dif_surface_mod", "home_bye_mod", "away_bye_mod", "home_net_qb_mod"]:
    d[c] = d[c].fillna(0)

print("=== (a) decomposition: nfelo_dif_base vs elo dif + hfa_mod + qb ===")
elo = d.starting_nfelo_home - d.starting_nfelo_away
r1 = d.nfelo_dif_base - (elo + d.hfa_mod + d.home_net_qb_mod)
r2 = d.hfa_mod - (d.hfa_base_mod + d.home_time_advantage_mod + d.div_game_mod + d.dif_surface_mod + d.home_bye_mod + d.away_bye_mod)
print("  nfelo_dif_base - (elo + hfa_mod + qb): mean %+.3f sd %.3f max|.| %.2f" % (r1.mean(), r1.std(), r1.abs().max()))
print("  hfa_mod - (base + tz + div + surf + hbye + abye): mean %+.4f sd %.4f max|.| %.4f" % (r2.mean(), r2.std(), r2.abs().max()))
print("  => hfa_mod is NOT a site HFA; it is the sum of ALL nfelo modifiers incl. the bye mods. ORIGINATOR's PFF/Cole HFA = (hfa_mod + tz)/25 = (base + 2*tz + div + surf + byes)/25")
print("  mean pts inside hfa_mod on one-bye games, bye-team perspective: %.2f" % (((d.home_bye_mod + d.away_bye_mod) / 25 * np.where(d.home_bye, 1, np.where(d.away_bye, -1, 0)))[d.home_bye ^ d.away_bye]).mean())
# ORIGINATOR proxy as specified: 0.46*nraw + 0.54*(-(elo + qb + hfa_mod + tz)/25) = nraw - 0.54*tz_pts
d["orig_as_built"] = d.nraw_line - 0.54 * d.nfelo_tz_pts
d["expert_proxy"] = 0.46 * d.nraw_line + 0.54 * d.rate_line
print("  as-built proxy minus expert proxy (pts, + = more toward away): mean %+.3f sd %.3f; on one-bye games bye-team perspective mean %+.3f" %
      ((d.orig_as_built - d.expert_proxy).mean(), (d.orig_as_built - d.expert_proxy).std(),
       (-(d.orig_as_built - d.expert_proxy) * np.where(d.home_bye, 1, -1))[d.home_bye ^ d.away_bye].mean()))

print("\n=== (b) 1999-2025 market-only sample (no nfelo needed) ===")
g = load_games(min_season=1999); g = g[(g.game_type == "REG") & g.mkt_spread.notna()].copy()
g["rv"] = g.week > 1
print("  rest NA by era:", g.groupby(g.season // 5 * 5).home_rest.apply(lambda s: s.isna().sum()).to_dict(), "| week1 rest values:", g[g.week == 1].home_rest.unique())
g = g[g.rv].copy()
hs, as_ = g.home_rest <= 5, g.away_rest <= 5
one = g[hs ^ as_].copy(); one["sgn"] = np.where(as_[one.index], 1, -1)
print("  asymmetric short-rest games 1999-2025: n=%d (by season: %s)" % (len(one), one.season.value_counts().sort_index().to_dict()))
n_, mu, se, p = desc(one.spread_err_mkt * one.sgn); w, l, ps, pct, pv = ats_side(one.spread_err_mkt, one.sgn)
print("  rested side vs market close: %+.2f (se %.2f, p=%.2f); ATS rested side %d-%d-%d (%.3f)" % (mu, se, p, w, l, ps, pct))
both = g[hs & as_]; sun = g[~hs & ~as_ & g.weekday.eq("Sunday")]
n1, mu1, se1, p1 = desc(both.total_err_mkt); n2, mu2, se2, p2 = desc(sun.total_err_mkt)
print("  TNF both-short 1999-2025: n=%d total resid %+.2f (se %.2f) vs Sunday n=%d %+.2f (se %.2f); diff %+.2f (Welch p=%.2f); over rate %.3f vs %.3f; spread |err| %.2f vs %.2f (Levene p=%.2f)" %
      (n1, mu1, se1, n2, mu2, se2, mu1 - mu2, stats.ttest_ind(both.total_err_mkt.dropna(), sun.total_err_mkt.dropna(), equal_var=False).pvalue,
       (both.total_err_mkt > 0).mean(), (sun.total_err_mkt > 0).mean(), both.spread_err_mkt.abs().mean(), sun.spread_err_mkt.abs().mean(),
       stats.levene(both.spread_err_mkt.dropna(), sun.spread_err_mkt.dropna()).pvalue))
# bye team vs market 1999-2025
hb, ab = (g.home_rest >= 13), (g.away_rest >= 13)
ob = g[hb ^ ab].copy(); ob["sgn"] = np.where(hb[ob.index], 1, -1)
n_, mu, se, p = desc(ob.spread_err_mkt * ob.sgn); w, l, ps, pct, pv = ats_side(ob.spread_err_mkt, ob.sgn)
print("  one-bye games 1999-2025: n=%d bye team vs close %+.2f (se %.2f p=%.2f); ATS %d-%d-%d %.3f" % (n_, mu, se, p, w, l, ps, pct))
for lo, hi in [(1999, 2008), (2009, 2025)]:
    x = ob[ob.season.between(lo, hi)]; n_, mu, se, p = desc(x.spread_err_mkt * x.sgn); print("    %d-%d n=%d %+.2f (se %.2f)" % (lo, hi, n_, mu, se))
# market residual ~ rest diff 1999-2008 (is the market's pricing 'right' pre-nfelo too?)
g["rd"] = (g.home_rest - g.away_rest).clip(-7, 7).astype(float)
for lo, hi in [(1999, 2008), (2009, 2025), (1999, 2025)]:
    x = g[g.season.between(lo, hi)]; co, r = ols(x.spread_err_mkt, [x.rd], ["rd"])
    print("  err_mkt ~ rest_diff %d-%d: slope %+.3f/day (se %.3f p=%.2f) n=%d" % (lo, hi, *co["rd"], len(x)))

print("\n=== (c) nfelo bye mod by season (pts for the bye team) ===")
dd = d[d.rest_valid & (d.home_bye ^ d.away_bye)].copy(); dd["sgn"] = np.where(dd.home_bye, 1, -1); dd["mod"] = dd.nfelo_bye_pts * dd.sgn
t = dd.groupby(["season", "sgn"])["mod"].mean().unstack().rename(columns={1: "home_bye", -1: "away_bye"}).round(2)
print(t.T.to_string())
print("  share of bye games with nfelo mod == 0 by era:", dd.groupby(dd.season // 4 * 4)["mod"].apply(lambda s: (s.abs() < 0.05).mean()).round(2).to_dict())

print("\n=== (d) does the linear term add beyond the bye dummy? residual ~ bye_sgn + rd_nonbye (HC1) ===")
dr = d[d.rest_valid].copy(); dr["rd"] = dr.rest_diff.clip(-7, 7).astype(float)
dr["bye_sgn"] = dr.home_bye.astype(float) - dr.away_bye.astype(float)
dr["rd_nb"] = np.where(dr.home_bye | dr.away_bye, 0.0, dr.rd)
for lab, x in [("FIT", dr[dr.fit]), ("TEST", dr[dr.test]), ("ALL", dr)]:
    for c in ["err_rate", "err_mkt"]:
        co, r = ols(x[c], [x.bye_sgn, x.rd_nb], ["bye_sgn", "rd_nonbye"])
        print("  %-4s %-8s bye=%+.2f (se %.2f p=%.2f)  rd_nonbye=%+.3f/day (se %.3f p=%.2f)  n=%d" % (lab, c, *co["bye_sgn"], *co["rd_nonbye"], len(x)))
    co, r = ols(x.mkt_spread, [x.rate_line, x.bye_sgn, x.rd_nb], ["rate_line", "bye", "rd_nb"])
    print("  %-4s market prices: bye %.2f (se %.2f), non-bye rest %.3f/day (se %.3f)" % (lab, -co["bye"][0], co["bye"][1], -co["rd_nb"][0], co["rd_nb"][1]))
print("  robust slopes of err_rate ~ rd (ALL, n=%d):" % len(dr))
X = sm.add_constant(dr.rd.values)
q = sm.QuantReg(dr.err_rate.values, X).fit(q=0.5); print("    LAD (median) slope %+.3f (se %.3f p=%.2f)" % (q.params[1], q.bse[1], q.pvalues[1]))
hb_ = sm.RLM(dr.err_rate.values, X, M=sm.robust.norms.HuberT()).fit(); print("    Huber slope %+.3f (se %.3f p=%.2f)" % (hb_.params[1], hb_.bse[1], hb_.pvalues[1]))
q2 = sm.QuantReg(dr.err_rate.values, sm.add_constant(dr.bye_sgn.values)).fit(q=0.5); print("    LAD bye-dummy effect %+.2f (se %.2f p=%.2f)" % (q2.params[1], q2.bse[1], q2.pvalues[1]))

print("\n=== (e) placebo: permute rest_diff within season (2000 draws) -> null for the err_rate slope; observed slope in ALL sample ===")
obs = ols(dr.err_rate, [dr.rd], ["rd"])[0]["rd"][0]
null = []
y = dr.err_rate.values; seasons = dr.season.values; rd = dr.rd.values.copy()
for i in range(2000):
    p_ = rd.copy()
    for s in np.unique(seasons):
        idx = np.where(seasons == s)[0]; p_[idx] = rng.permutation(rd[idx])
    null.append(np.polyfit(p_, y, 1)[0])
null = np.array(null); print("  observed %+.3f | permutation null sd %.3f | two-sided p = %.3f" % (obs, null.std(), (np.abs(null) >= abs(obs)).mean()))
# 'future rest' placebo: use each team's NEXT game's rest differential (not causal for this game) as a regressor
gg = m[m.game_type.eq("REG")].sort_values(["season", "week"]).copy()
def next_rest(team_col, rest_col):
    out = pd.Series(np.nan, index=gg.index)
    for team in pd.unique(pd.concat([gg.home, gg.away])):
        idx = gg.index[(gg.home == team) | (gg.away == team)]
        rests = np.where(gg.loc[idx, "home"] == team, gg.loc[idx, "home_rest"], gg.loc[idx, "away_rest"]).astype(float)
        nxt = np.r_[rests[1:], np.nan]
        out.loc[idx] = np.where(gg.loc[idx, team_col] == team, nxt, out.loc[idx])
    return out
gg["home_next_rest"] = next_rest("home", "home_rest"); gg["away_next_rest"] = next_rest("away", "away_rest")
gg["rd_next"] = (gg.home_next_rest - gg.away_next_rest).clip(-7, 7)
x = gg.dropna(subset=["rd_next", "nfelo_dif_base"]); x = x[x.rest_valid]
co, r = ols(x.err_rate, [x.rd_next], ["rd_next"]); print("  placebo: err_rate ~ NEXT game's rest diff: slope %+.3f (se %.3f p=%.2f) n=%d" % (*co["rd_next"], len(x)))

print("\n=== (f) rolling-origin (fit on seasons < s, test s), affected games: bias reduction with paired bootstrap CI ===")
rows = []
for s in range(2016, 2026):
    f = dr[dr.season < s]; t = dr[dr.season == s]
    ks = np.arange(0, 0.51, 0.025); k = ks[int(np.argmin([(f.margin + f.rate_line - kk * f.rd).abs().mean() for kk in ks]))]
    a = t[t.rd != 0]; e0 = (a.margin + a.rate_line) * np.sign(a.rd); e1 = e0 - k * a.rd.abs()
    rows.append(pd.DataFrame({"season": s, "e0": e0.values, "e1": e1.values, "ae0": (a.margin + a.rate_line).abs().values, "ae1": (a.margin + a.rate_line - k * a.rd).abs().values}))
R = pd.concat(rows); n_ = len(R)
bs_b = [(R.e1.values[i] - R.e0.values[i]).mean() for i in (rng.integers(0, n_, n_) for _ in range(4000))]
bs_m = [(R.ae1.values[i] - R.ae0.values[i]).mean() for i in (rng.integers(0, n_, n_) for _ in range(4000))]
print("  n=%d rested-side bias %+.3f -> %+.3f (change %+.3f, 95%% CI [%+.3f, %+.3f]) | MAE %.4f -> %.4f (change %+.4f, CI [%+.4f, %+.4f])" %
      (n_, R.e0.mean(), R.e1.mean(), R.e1.mean() - R.e0.mean(), np.percentile(bs_b, 2.5), np.percentile(bs_b, 97.5), R.ae0.mean(), R.ae1.mean(), R.ae1.mean() - R.ae0.mean(), np.percentile(bs_m, 2.5), np.percentile(bs_m, 97.5)))
print("  se of the rested-side bias itself on these games: %.3f (so the base bias %+.3f has p=%.3f)" % (R.e0.std() / np.sqrt(n_), R.e0.mean(), stats.ttest_1samp(R.e0, 0).pvalue))

print("\n=== (g) bye-team bias on the ORIGINATOR proxy AS BUILT (README: PFF/Cole HFA = (hfa_mod + tz)/25, i.e. nfelo's bye mod is ALSO inside the 54%% share) ===")
for lab, x in [("ALL 2009-25", dd), ("FIT", dd[dd.fit]), ("TEST 2022-25", dd[dd.test])]:
    base_built = x.nraw_line - 0.54 * x.nfelo_tz_pts
    spec_built = base_built - 0.75 * x.sgn
    expert_base = 0.46 * x.nraw_line + 0.54 * x.rate_line
    cand = x.rate_line - 1.0 * x.sgn          # strip everything, apply 1.0 once
    out = []
    for nm, line in [("as-built, no spec", base_built), ("as-built + spec +0.75", spec_built), ("expert's proxy (a)", expert_base), ("candidate strip+1.0", cand), ("market", x.mkt_spread)]:
        e = (x.margin + line) * x.sgn; out.append("%s %+.2f" % (nm, e.mean()))
    print("  %-12s n=%3d bye-team bias: %s | (se ~%.2f) | total bye pts carried as-built+spec: %.2f" % (lab, len(x), " | ".join(out), (x.err_rate * x.sgn).std() / np.sqrt(len(x)), (x["mod"] + 0.75).mean()))
print("  NOTE the expert's text says 'subtract (home_bye_mod + away_bye_mod)/25 from its line' to strip nfelo's bye mods; in ORIGINATOR sign (neg = home fav) the strip is line + (home_bye_mod + away_bye_mod)/25 (as common.py does). Literal implementation would double the mod.")
