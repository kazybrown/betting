"""CRITIC 05 / Theories 5a (W1 HFA) and 5b (W1 information beyond ratings -> caps).
5a attacks: reproduce; season-clustered SE; a 1999-2025 version using the naive prev-season-MOV rating (n~420 W1 games)
    so the W1 HFA intercept has 27 seasons instead of 17; market home residual W1 by era with clustered SE.
5b attacks: reproduce the info-share regression; season-clustered SEs; per-week shares 2009-14 vs 2015-25;
    drop W3 (the 0.25 outlier) from the W1-4 pool; use nraw (WITH the QB adjustment) as the rating line;
    nonlinearity (|dev|<=2.5 vs >2.5); link to 4b: add the net new-QB flag to the W1 regression;
    rolling-origin cap test 2015-25 (cap chosen on seasons < t) for W1 / W1-4 / W5+; and a placebo where 'dev' is the
    market's deviation from the nfelo CLOSE (which is already ~0.7 regressed to the market).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import build, desc, ols, paired_mae

pd.set_option("display.width", 250)
m = build(min_season=1999)
d = m[m.nfelo_dif_base.notna() & ~m.neutral].copy(); d["dev"] = d.mkt - d.elo_line; d["dev_raw"] = d.mkt - d.nraw


def fit(y, X, groups=None):
    Xm = sm.add_constant(np.column_stack(X))
    if groups is None: return sm.OLS(np.asarray(y, float), Xm).fit(cov_type="HC1")
    return sm.OLS(np.asarray(y, float), Xm).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})


print("5a. Week-1 HFA")
w1f = (d.week == 1).astype(float)
for lab, g in [("HC1", None), ("cluster(season)", d.season)]:
    r = fit(d.margin, [d.elo_dif_pts, w1f, d.elo_dif_pts * w1f], g)
    print("  2009-25 non-neutral n=%d %-16s HFA other %+.2f (se %.2f) | W1 shift %+.2f (se %.2f, p=%.3f)" % (len(d), lab, r.params[0], r.bse[0], r.params[2], r.bse[2], r.pvalues[2]))
# 1999-2025 with naive prev-MOV rating: margin ~ a + b*pmov_dif + W1 + W1*pmov_dif
h = m[["season", "home", "margin"]].rename(columns={"home": "team"}); h["mov"] = h.margin
a = m[["season", "away", "margin"]].rename(columns={"away": "team"}); a["mov"] = -a.margin
tm = pd.concat([h[["season", "team", "mov"]], a[["season", "team", "mov"]]]).groupby(["season", "team"]).mov.mean().rename("pmov").reset_index(); tm["season"] += 1
mm = m.merge(tm.rename(columns={"team": "home", "pmov": "hp"}), on=["season", "home"]).merge(tm.rename(columns={"team": "away", "pmov": "ap"}), on=["season", "away"])
mm = mm[~mm.neutral].copy(); mm["pd"] = mm.hp - mm.ap; w1n = (mm.week == 1).astype(float); e4 = (mm.week <= 4).astype(float)
r = fit(mm.margin, [mm.pd, w1n, mm.pd * w1n], mm.season)
print("  2000-25 naive-rating n=%d cluster(season): HFA other %+.2f (se %.2f) | W1 shift %+.2f (se %.2f, p=%.3f)" % (len(mm), r.params[0], r.bse[0], r.params[2], r.bse[2], r.pvalues[2]))
r = fit(mm.margin, [mm.pd, e4, mm.pd * e4], mm.season)
print("  2000-25 naive-rating: W1-4 shift %+.2f (se %.2f, p=%.3f)" % (r.params[2], r.bse[2], r.pvalues[2]))
x = m[(m.week == 1) & ~m.neutral]
for lab, mask in [("1999-2025", x.season >= 1999), ("1999-2012", x.season <= 2012), ("2013-2025", x.season >= 2013), ("2021-2025", x.season >= 2021)]:
    z = x[mask]; r = fit(z.err_mkt, [np.zeros(len(z))], z.season)
    print("  market W1 home residual %-9s %+.2f (cluster se %.2f, p=%.3f, n=%d)" % (lab, r.params[0], r.bse[0], r.pvalues[0], len(z)))
# W1 vs rest for the market home residual, clustered
xx = m[~m.neutral]; r = fit(xx.err_mkt, [(xx.week == 1).astype(float)], xx.season)
print("  market home residual W1 minus other weeks (1999-2025, cluster): %+.2f (se %.2f, p=%.3f)" % (r.params[1], r.bse[1], r.pvalues[1]))

print("\n5b. Information share of the market's deviation from the rating line (-slope of err_elo_line on dev)")
print("  A. reproduce + clustered")
for lab, g in [("HC1", None), ("cluster(season)", d.season)]:
    r = fit(d.err_elo_line, [d.dev, w1f, d.dev * w1f], g); e4 = (d.week <= 4).astype(float); r4 = fit(d.err_elo_line, [d.dev, e4, d.dev * e4], g)
    print("    %-16s other %.2f (se %.2f) W1 change %+.2f (se %.2f, p=%.3f) | W5+ %.2f W1-4 change %+.2f (se %.2f, p=%.3f)" %
          (lab, -r.params[1], r.bse[1], -r.params[3], r.bse[3], r.pvalues[3], -r4.params[1], -r4.params[3], r4.bse[3], r4.pvalues[3]))
print("  B. per-week share by era (HC1):")
for lab, mask in [("2009-2014", d.season <= 2014), ("2015-2025", d.season >= 2015), ("2009-2025", d.season >= 2009)]:
    out = []
    for w in ["1", "2", "3", "4", "5-9", "10+"]:
        x = d[mask & (d.wk == w)]; r = fit(x.err_elo_line, [x.dev]); out.append("W%s %.2f (%.2f)" % (w, -r.params[1], r.bse[1]))
    print("    %-9s " % lab + " | ".join(out))
print("  C. W1-4 pooled test WITHOUT week 3, and W1+W2 only, and W2-4 only (2009-25):")
for lab, early_mask in [("W1,2,4 vs W5+", d.week.isin([1, 2, 4])), ("W1-2 vs W5+", d.week <= 2), ("W2-4 vs W5+", d.week.between(2, 4))]:
    x = d[early_mask | (d.week >= 5)]; e = early_mask[early_mask | (d.week >= 5)].astype(float)
    r = fit(x.err_elo_line, [x.dev, e, x.dev * e], x.season)
    print("    %-14s early share %.2f | change %+.2f (cluster se %.2f, p=%.3f)" % (lab, -(r.params[1] + r.params[3]), -r.params[3], r.bse[3], r.pvalues[3]))
print("  D. rating line WITH the QB adjustment (nraw): dev_raw = mkt - nraw")
for lab, mask in [("2009-2025", d.season >= 2009), ("2015-2025", d.season >= 2015)]:
    x = d[mask]; w = (x.week == 1).astype(float); e = (x.week <= 4).astype(float)
    r = fit(x.err_nraw, [x.dev_raw, w, x.dev_raw * w], x.season); r4 = fit(x.err_nraw, [x.dev_raw, e, x.dev_raw * e], x.season)
    print("    %-9s other %.2f W1 %.2f (change %+.2f, se %.2f, p=%.3f) | W5+ %.2f W1-4 %.2f (change %+.2f, se %.2f, p=%.3f)" %
          (lab, -r.params[1], -(r.params[1] + r.params[3]), -r.params[3], r.bse[3], r.pvalues[3], -r4.params[1], -(r4.params[1] + r4.params[3]), -r4.params[3], r4.bse[3], r4.pvalues[3]))
print("  E. nonlinearity: W1 share for |dev|<=2.5 vs >2.5, and W10+")
for w in ["1", "1-4", "10+"]:
    x = d[(d.week == 1) if w == "1" else (d.week <= 4) if w == "1-4" else (d.week >= 10)]
    for lab, mask in [("|dev|<=2.5", x.dev.abs() <= 2.5), ("|dev|>2.5", x.dev.abs() > 2.5)]:
        z = x[mask]; r = fit(z.err_elo_line, [z.dev])
        print("    wk %-4s %-10s n=%4d share %.2f (se %.2f)" % (w, lab, len(z), -r.params[1], r.bse[1]))
print("  F. link to theory 4b: W1 share with the net new-QB flag as a covariate (flags from critic_04 construction)")
hq = m[["season", "week", "home", "home_qb_name"]].rename(columns={"home": "team", "home_qb_name": "qb"}); aq = m[["season", "week", "away", "away_qb_name"]].rename(columns={"away": "team", "away_qb_name": "qb"})
tq = pd.concat([hq, aq]).sort_values(["team", "season", "week"]); rows = []
for (team, season), x in tq.groupby(["team", "season"]):
    prev = tq[(tq.team == team) & (tq.season == season - 1)]
    if len(prev) == 0: continue
    rows.append(dict(team=team, season=season, new_qb=int(x.sort_values("week").qb.iloc[0] != prev.qb.mode().iloc[0])))
ts = pd.DataFrame(rows)
d = d.merge(ts.rename(columns={"team": "home", "new_qb": "hq"}), on=["season", "home"], how="left").merge(ts.rename(columns={"team": "away", "new_qb": "aq"}), on=["season", "away"], how="left")
d["net_q"] = d.hq - d.aq
x = d[(d.week == 1) & d.net_q.notna()]
r0 = fit(x.err_elo_line, [x.dev]); r1 = fit(x.err_elo_line, [x.dev, x.net_q]); r2 = fit(x.dev, [x.net_q])
print("    W1 n=%d share alone %.2f (se %.2f) | with new-QB flag: share %.2f (se %.2f), flag coef %+.2f (se %.2f) | market prices new-QB flag at dev %+.2f (se %.2f)" %
      (len(x), -r0.params[1], r0.bse[1], -r1.params[1], r1.bse[1], r1.params[2], r1.bse[2], r2.params[1], r2.bse[1]))
x2 = x[x.net_q == 0]; r = fit(x2.err_elo_line, [x2.dev]); print("    W1 games with NO new-QB on either side: n=%d share %.2f (se %.2f)" % (len(x2), -r.params[1], r.bse[1]))
x3 = x[x.net_q != 0]; r = fit(x3.err_elo_line, [x3.dev]); print("    W1 games with a new-QB flag: n=%d share %.2f (se %.2f)" % (len(x3), -r.params[1], r.bse[1]))

print("  G. rolling-origin cap test 2015-25: cap c chosen on seasons < t (grid 1,1.5,2,2.5,3,4,none) by MAE, applied to season t; paired MAE vs market")
caps = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99]
for lab, wmask in [("W1", d.week == 1), ("W1-4", d.week <= 4), ("W2-4", d.week.between(2, 4)), ("W5+", d.week >= 5)]:
    ec, em, chosen = [], [], []
    for t in range(2015, 2026):
        tr = d[(d.season < t) & wmask]; te = d[(d.season == t) & wmask]
        fm = {c: (tr.margin + tr.elo_line + tr.dev.clip(-c, c)).abs().mean() for c in caps}; c = min(fm, key=fm.get); chosen.append(c)
        ec.append(te.margin + te.elo_line + te.dev.clip(-c, c)); em.append(te.err_mkt)
    ec, em = pd.concat(ec), pd.concat(em); dd, lo, hi, p, n = paired_mae(ec, em)
    fixed = []
    for c in (1.0, 1.5, 2.5):
        e = pd.concat([d[(d.season == t) & wmask].margin + d[(d.season == t) & wmask].elo_line + d[(d.season == t) & wmask].dev.clip(-c, c) for t in range(2015, 2026)])
        dk, lok, hik, pk, _ = paired_mae(e, em); fixed.append("c=%.1f %+.3f [%+.2f,%+.2f]" % (c, dk, lok, hik))
    print("    %-5s n=%4d chosen caps %s | MAE capped %.3f mkt %.3f diff %+.3f [%+.2f,%+.2f] p=%.3f | fixed: %s" % (lab, n, sorted(set(chosen)), ec.abs().mean(), em.abs().mean(), dd, lo, hi, p, "; ".join(fixed)))

print("  H. Placebo: 'dev' measured from the nfelo CLOSE (already regressed to market) — share should be ~1 everywhere if the method is sound")
d["dev_c"] = d.mkt - d.nclose
for w in ["1", "1-4", "5+"]:
    x = d[(d.week == 1) if w == "1" else (d.week <= 4) if w == "1-4" else (d.week >= 5)].dropna(subset=["dev_c"]); r = fit(x.err_nclose, [x.dev_c])
    print("    wk %-4s n=%4d share of (mkt - nfelo close) %.2f (se %.2f) | SD dev_c %.2f" % (w, len(x), -r.params[1], r.bse[1], x.dev_c.std()))
