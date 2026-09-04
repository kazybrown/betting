"""CRITIC 04 / Theories 4a (new HC) + 4b (new W1 starting QB) — the only SUPPORTED spread adjustment.
Flags rebuilt exactly as in the expert's 04 script.  Attacks on 4b:
  1. Reproduce W1 coef (+2.13) and ATS.
  2. Decompose new_qb into 'QB new to team' (no start for team last year) vs 'took over during last season'.
  3. Confounders: prev-season MOV of each team, market line, new HC, div game; home-only / away-only.
  4. Season-clustered SE; leave-one-season-out range; per-era; per-season sign count.
  5. Placebo by week: the flag is a season attribute -> W2, W3, W4, W5-8 separately.
  6. Alternative definitions: (a) starter != last season's FINAL-game starter; (b) first-ever start (rookie / never started).
  7. Multiplicity: the expert ran 5 flags x 3 windows x (coef, ATS); Bonferroni on the headline p.
  8. OOS value of the recommended +1.0: shift the MARKET line 1.0 toward the flagged team; paired MAE vs market,
     all W1 and rolling by era; plus ATS of the +1.0-adjusted number.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, desc, binom, ols, paired_mae

pd.set_option("display.width", 250)
m = build(min_season=1999)
h = m[["season", "week", "home", "home_coach", "home_qb_name", "margin"]].rename(columns={"home": "team", "home_coach": "coach", "home_qb_name": "qb"}); h["mov"] = h.margin
a = m[["season", "week", "away", "away_coach", "away_qb_name", "margin"]].rename(columns={"away": "team", "away_coach": "coach", "away_qb_name": "qb"}); a["mov"] = -a.margin
tg = pd.concat([h, a]).sort_values(["team", "season", "week"])
rows = []
for (team, season), x in tg.groupby(["team", "season"]):
    prev = tg[(tg.team == team) & (tg.season == season - 1)].sort_values("week")
    w1 = x[x.week == x.week.min()]
    if len(w1) == 0 or len(prev) == 0: continue
    coach1, qb1 = w1.coach.iloc[0], w1.qb.iloc[0]
    prev_primary_qb = prev.qb.mode().iloc[0]; prev_last_qb = prev.qb.dropna().iloc[-1] if prev.qb.notna().any() else np.nan
    prev_last3 = prev.qb.dropna().iloc[-3:].mode(); prev_last3 = prev_last3.iloc[0] if len(prev_last3) else np.nan
    started_before = qb1 in set(tg[(tg.season < season)].qb.dropna())
    rows.append(dict(team=team, season=season, new_hc=int(coach1 not in set(prev.coach.dropna())),
                     new_qb=int(qb1 != prev_primary_qb), qb_new_to_team=int(qb1 not in set(prev.qb.dropna())),
                     qb_took_over=int((qb1 != prev_primary_qb) and (qb1 in set(prev.qb.dropna()))),
                     new_vs_last=int(qb1 != prev_last_qb), new_vs_last3=int(qb1 != prev_last3),
                     first_start=int((qb1 != prev_primary_qb) and (not started_before)) if season >= 2001 else np.nan,
                     vet_new=int((qb1 != prev_primary_qb) and started_before) if season >= 2001 else np.nan,
                     prev_mov=prev.mov.mean()))
ts = pd.DataFrame(rows)
print("team-seasons %d | new_qb %d = qb_new_to_team %d + qb_took_over %d | new_vs_last %d | first_start %d vet_new %d" %
      (len(ts), ts.new_qb.sum(), ts.qb_new_to_team.sum(), ts.qb_took_over.sum(), ts.new_vs_last.sum(), ts.first_start.sum(), ts.vet_new.sum()))
flags = ["new_hc", "new_qb", "qb_new_to_team", "qb_took_over", "new_vs_last", "new_vs_last3", "first_start", "vet_new", "prev_mov"]
for side in ["home", "away"]:
    m = m.merge(ts[["team", "season"] + flags].rename(columns={"team": side, **{c: f"{side}_{c}" for c in flags}}), on=[side, "season"], how="left")
for f in flags:
    m[f"net_{f}"] = m[f"home_{f}"] - m[f"away_{f}"]
w1 = m[(m.week == 1) & m.home_new_qb.notna() & m.away_new_qb.notna()].copy()
print("W1 games with flags: %d (1999-2025)" % len(w1))


def reg(df, y, xs, label, cluster=False):
    dd = df.dropna(subset=[y] + xs)
    X = sm.add_constant(np.column_stack([dd[c].values for c in xs]))
    if cluster:
        r = sm.OLS(dd[y].values, X).fit(cov_type="cluster", cov_kwds={"groups": dd.season.values})
    else:
        r = sm.OLS(dd[y].values, X).fit(cov_type="HC1")
    ev = int((dd[xs[0]] != 0).sum())
    print("  %-58s n=%4d ev=%3d  " % (label, len(dd), ev) + " | ".join("%s %+.2f (se %.2f, p=%.3f)" % (c, r.params[i + 1], r.bse[i + 1], r.pvalues[i + 1]) for i, c in enumerate(xs)))
    return r


def ats_back(df, x, label):
    dd = df[df[x].notna() & (df[x] != 0)]
    win = ((dd[x] > 0) & (dd.err_mkt > 0)) | ((dd[x] < 0) & (dd.err_mkt < 0)); push = dd.err_mkt == 0
    w = int(win.sum()); l = int((~win & ~push).sum()); pct, lo, hi, p = binom(w, l)
    print("  %-58s back-flag ATS %d-%d-%d = %.3f [%.3f,%.3f] p=%.3f" % (label, w, l, int(push.sum()), pct, lo, hi, p))


print("\n1. Reproduce W1 new_qb (vs prev primary), residual vs market, HC1 and season-clustered")
reg(w1, "err_mkt", ["net_new_qb"], "new_qb HC1"); reg(w1, "err_mkt", ["net_new_qb"], "new_qb cluster(season)", cluster=True); ats_back(w1, "net_new_qb", "new_qb")
reg(w1, "err_mkt", ["net_new_hc"], "new_hc HC1 (4a)"); ats_back(w1, "net_new_hc", "new_hc")

print("\n2. Decomposition of new_qb")
reg(w1, "err_mkt", ["net_qb_new_to_team", "net_qb_took_over"], "new_to_team + took_over (jointly)")
ats_back(w1, "net_qb_new_to_team", "QB new to team"); ats_back(w1, "net_qb_took_over", "QB took over last season (not primary)")
reg(w1, "err_mkt", ["net_first_start", "net_vet_new"], "first-ever start + veteran new starter (2001+)")
ats_back(w1, "net_first_start", "first-ever start"); ats_back(w1, "net_vet_new", "veteran new starter")

print("\n3. Confounders")
reg(w1, "err_mkt", ["net_new_qb", "net_prev_mov"], "+ net prev-season MOV")
reg(w1, "err_mkt", ["net_new_qb", "net_prev_mov", "mkt"], "+ prev MOV + market line")
reg(w1, "err_mkt", ["net_new_qb", "net_new_hc", "net_prev_mov", "mkt"], "+ new_hc + prev MOV + line")
print("  correlation net_new_qb vs net_prev_mov = %.3f, vs mkt = %.3f" % (np.corrcoef(w1.net_new_qb, w1.net_prev_mov)[0, 1], np.corrcoef(w1.net_new_qb, w1.mkt)[0, 1]))
reg(w1, "err_mkt", ["home_new_qb"], "home flag only"); reg(w1, "err_mkt", ["away_new_qb"], "away flag only (expect negative)")
for lab, mask in [("flagged team is dog", None)]:
    pass
# team-level: flagged team as dog vs fav
tv = pd.concat([w1[w1.home_new_qb == 1].assign(cvr=lambda z: z.err_mkt, tline=lambda z: z.mkt), w1[w1.away_new_qb == 1].assign(cvr=lambda z: -z.err_mkt, tline=lambda z: -z.mkt)])
for lab, mask in [("flagged team underdog (tline>0)", tv.tline > 0), ("flagged team favourite (tline<0)", tv.tline < 0)]:
    z = tv[mask]; n, mu, se, p = desc(z.cvr); print("  %-40s n=%3d cover margin %+.2f (se %.2f) p=%.3f | ATS %d-%d" % (lab, n, mu, se, p, int((z.cvr > 0).sum()), int((z.cvr < 0).sum())))

print("\n4. Stability: per era, leave-one-season-out, per-season team-level sign")
for lab, mask in [("1999-2008", w1.season <= 2008), ("2009-2016", w1.season.between(2009, 2016)), ("2017-2025", w1.season >= 2017), ("2000-2012", w1.season <= 2012), ("2013-2025", w1.season >= 2013)]:
    reg(w1[mask], "err_mkt", ["net_new_qb"], "new_qb " + lab)
loo = []
for s in sorted(w1.season.unique()):
    dd = w1[w1.season != s]; X = sm.add_constant(dd.net_new_qb.values); loo.append(sm.OLS(dd.err_mkt.values, X).fit().params[1])
print("  leave-one-season-out coef range %.2f .. %.2f (mean %.2f)" % (min(loo), max(loo), np.mean(loo)))
tv2 = tv.groupby("season").cvr.mean()
print("  per-season mean cover margin of flagged teams: positive in %d of %d seasons | mean of season means %+.2f (se %.2f, p=%.3f)" % (int((tv2 > 0).sum()), len(tv2), tv2.mean(), tv2.std() / np.sqrt(len(tv2)), stats.ttest_1samp(tv2, 0).pvalue))
print("  " + ", ".join("%d:%+.1f" % (s, v) for s, v in tv2.items()))
print("  median team-level cover margin %+.1f | Wilcoxon p=%.3f | 10%%-trimmed mean %+.2f" % (tv.cvr.median(), stats.wilcoxon(tv.cvr[tv.cvr != 0]).pvalue, stats.trim_mean(tv.cvr, 0.1)))

print("\n5. Placebo by week (same season-level flag)")
for lab, mask in [("W1", m.week == 1), ("W2", m.week == 2), ("W3", m.week == 3), ("W4", m.week == 4), ("W2-4", m.week.between(2, 4)), ("W5-8", m.week.between(5, 8)), ("W9-18", m.week >= 9)]:
    x = m[mask & m.home_new_qb.notna() & m.away_new_qb.notna()]
    reg(x, "err_mkt", ["net_new_qb"], "new_qb " + lab); 

print("\n6. Alternative definitions (W1)")
reg(w1, "err_mkt", ["net_new_vs_last"], "starter != last season's FINAL-game starter"); ats_back(w1, "net_new_vs_last", "vs final-game starter")
reg(w1, "err_mkt", ["net_new_vs_last3"], "starter != mode of last season's last 3 starters"); ats_back(w1, "net_new_vs_last3", "vs last-3 mode")
reg(w1, "err_mkt", ["net_qb_new_to_team"], "QB new to team (expert alt)")

print("\n7. Multiplicity: headline p=0.029; expert tested 5 flags x {ALL,FIT,TEST} x {coef, ATS} in W1 plus 3 windows -> >=15 tests. Bonferroni(15) p = %.2f; Bonferroni(5 flags) p = %.2f" % (min(1, 0.029 * 15), min(1, 0.029 * 5)))

print("\n8. OOS value of the recommended +1.0 lean: market line shifted 1.0 toward the flagged team (net flag), paired MAE vs market")
w1["adj_line"] = w1.mkt - 1.0 * w1.net_new_qb          # more negative = home favored more; flag on home -> lean home
for lab, mask in [("ALL 1999-2025", w1.season >= 1999), ("FIT 1999-2021", w1.season <= 2021), ("TEST 2022-2025", w1.season >= 2022), ("2009-2025", w1.season >= 2009), ("2015-2025", w1.season >= 2015)]:
    x = w1[mask & (w1.net_new_qb != 0)]
    dd, lo, hi, p, n = paired_mae(x.margin + x.adj_line, x.err_mkt)
    for k in (1.0, 2.0):
        e = x.margin + x.mkt - k * x.net_new_qb; dk, lok, hik, pk, _ = paired_mae(e, x.err_mkt)
        print("  %-15s n=%3d shift %.1f: MAE adj %.3f vs mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f" % (lab, n, k, e.abs().mean(), x.err_mkt.abs().mean(), dk, lok, hik, pk))
