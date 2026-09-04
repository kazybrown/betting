"""THEORY 4: dome / closed-roof effect on totals after controlling for the teams.
Identification: team-season fixed effects for BOTH home and away teams (offense+defense
level of each team that year), so the roof effect comes from the same teams playing in
different roof conditions (dome teams away outdoors, outdoor teams visiting domes).
Questions: (a) raw effect of dome/closed vs outdoors (b) vs outdoors in GOOD weather
(wind<10, temp>=50) -> is there a dome effect beyond weather? (c) does the MARKET already
price it (market residual by roof)? (d) OOS: adding a dome term to the LEAN model.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from common import build, mae, paired_mae_ci, ou_rate

m = build(K_team=1, K_lg=128, verbose=False)
m = m[(m.game_type == "REG") & m.lg_prev.notna() & ~m.neutral].copy()
m["dome"] = m.is_dome.astype(int)
m["roof3"] = np.where(m.roof == "dome", "dome", np.where(m.roof == "closed", "closed", np.where(m.roof == "open", "open", "outdoors")))
m["good_wx"] = ((m.outdoor == 1) & (m.wind < 10) & (m.temp >= 50)).astype(int)
m["bad_wx"] = ((m.outdoor == 1) & ((m.wind >= 15) | (m.temp < 32))).astype(int)
m["y"] = m.total_pts - m.lg_blend
m["hts"] = m.home + "_" + m.season.astype(str); m["ats"] = m.away + "_" + m.season.astype(str)

print("== (c) market residual by roof (does the market price the roof?) ==")
for lab, d in [("ALL 1999-2025", m), ("FIT <=2021", m[m.train]), ("TEST 2022-25", m[m.test])]:
    print(f"-- {lab}")
    for r in ["outdoors", "open", "closed", "dome"]:
        x = d[d.roof3 == r]
        print(f"    {r:9s} n={len(x):5d} total={x.total_pts.mean():.2f} mkt={x.mkt_total.mean():.2f} residual={x.total_err_mkt.mean():+.2f} (se {x.total_err_mkt.std()/np.sqrt(max(len(x),1)):.2f}) over-rate={(x.total_err_mkt>0).sum()/max((x.total_err_mkt!=0).sum(),1):.3f}")
    a = d[d.dome == 1]; b = d[d.dome == 0]
    t = stats.ttest_ind(a.total_err_mkt, b.total_err_mkt, equal_var=False)
    print(f"    dome/closed vs outdoor market residual diff={a.total_err_mkt.mean()-b.total_err_mkt.mean():+.2f} p={t.pvalue:.3f}")


def participant_fe_ols(d, xcols):
    """OLS of y on xcols + ONE effect per team-season shared by home and away appearances
    (venue-agnostic participant effects). No intercept (the FE absorb the level). HC1 SEs.
    Returns (params, bse, pvalues) for xcols only."""
    ts = sorted(set(d.hts) | set(d.ats)); idx = {t: i for i, t in enumerate(ts)}
    F = np.zeros((len(d), len(ts)))
    F[np.arange(len(d)), d.hts.map(idx).values] += 1.0
    F[np.arange(len(d)), d.ats.map(idx).values] += 1.0
    X = np.hstack([d[xcols].astype(float).values, F])
    r = sm.OLS(d.y.values.astype(float), X).fit(cov_type="HC1")
    k = len(xcols)
    return dict(zip(xcols, r.params[:k])), dict(zip(xcols, r.bse[:k])), dict(zip(xcols, r.pvalues[:k]))


print("\n== (a) raw dome effect with PARTICIPANT team-season fixed effects (one effect per team-season, home or away), y = total - league prior ==")
m["r_dome"] = (m.roof3 == "dome").astype(int); m["r_closed"] = (m.roof3 == "closed").astype(int); m["r_open"] = (m.roof3 == "open").astype(int)
for lab, d in [("ALL 1999-2025", m), ("1999-2010", m[m.season <= 2010]), ("2011-2021", m[(m.season > 2010) & (m.season <= 2021)]), ("FIT <=2021", m[m.train]), ("TEST 2022-2025", m[m.test])]:
    b, se, pv = participant_fe_ols(d, ["dome"])
    b3, se3, pv3 = participant_fe_ols(d, ["r_dome", "r_closed", "r_open"])
    print(f"  {lab:14s} n={len(d):5d} dome/closed={b['dome']:+.2f} (se {se['dome']:.2f}, p={pv['dome']:.3f}) | by type vs outdoors: dome={b3['r_dome']:+.2f}({se3['r_dome']:.2f}) closed={b3['r_closed']:+.2f}({se3['r_closed']:.2f}) open={b3['r_open']:+.2f}({se3['r_open']:.2f})")
f = smf.ols("y ~ dome", data=m).fit(cov_type="HC1")
print(f"  no team control, ALL: dome={f.params['dome']:+.2f} (se {f.bse['dome']:.2f})  <- raw difference includes team quality (dome teams' offenses)")

print("\n== (b) dome vs outdoor GOOD weather (wind<10 & temp>=50) vs MID vs BAD (wind>=15 or temp<32), participant team-season FE ==")
d = m[m.outdoor.eq(0) | m.wind.notna()].copy()   # outdoor rows need observed weather to classify
d["cond"] = np.where(d.dome == 1, "dome", np.where(d.good_wx == 1, "out_good", np.where(d.bad_wx == 1, "out_bad", "out_mid")))
d["c_dome"] = (d.cond == "dome").astype(int); d["c_mid"] = (d.cond == "out_mid").astype(int); d["c_bad"] = (d.cond == "out_bad").astype(int)
for lab, dd in [("ALL", d), ("FIT <=2021", d[d.train]), ("TEST 2022-25", d[d.test])]:
    b, se, pv = participant_fe_ols(dd, ["c_dome", "c_mid", "c_bad"])
    print(f"  {lab:12s} n={len(dd):5d} vs outdoor-good: dome={b['c_dome']:+.2f} (se {se['c_dome']:.2f}, p={pv['c_dome']:.3f})  out_mid={b['c_mid']:+.2f} (se {se['c_mid']:.2f})  out_bad={b['c_bad']:+.2f} (se {se['c_bad']:.2f})  "
          f"| counts: {dd.cond.value_counts().to_dict()}")
print("  market residual by condition (ALL):")
for c in ["dome", "out_good", "out_mid", "out_bad"]:
    x = d[d.cond == c]
    print(f"    {c:9s} n={len(x):5d} residual={x.total_err_mkt.mean():+.2f} (se {x.total_err_mkt.std()/np.sqrt(len(x)):.2f})")

print("\n== venue or home team? dome x (away team is a dome team), participant FE ==")
home_dome_share = m.groupby(["home", "season"]).dome.mean().rename("h_dome_share").reset_index()
m2 = m.merge(home_dome_share.rename(columns={"home": "away", "h_dome_share": "away_is_dome_team"}), on=["away", "season"], how="left")
m2["away_dome_team"] = (m2.away_is_dome_team > 0.5).astype(int)
m2["dome_x_awaydome"] = m2.dome * m2.away_dome_team
b, se, pv = participant_fe_ols(m2, ["dome", "dome_x_awaydome"])
print(f"  dome (outdoor visitor)={b['dome']:+.2f} (se {se['dome']:.2f})  extra when visitor is also a dome team={b['dome_x_awaydome']:+.2f} (se {se['dome_x_awaydome']:.2f})")
# by era, dome effect controlling for wind linearly (so 'dome' = vs a zero-wind outdoor game) and vs mean-wind outdoor game
d["wind_o"] = np.where(d.outdoor == 1, d.wind, 0.0); d["cold20"] = ((d.outdoor == 1) & (d.temp < 20)).astype(int)
for lab, dd in [("ALL", d), ("FIT <=2021", d[d.train])]:
    b, se, pv = participant_fe_ols(dd, ["dome", "wind_o", "cold20"])
    mw = dd.loc[dd.outdoor == 1, "wind"].mean()
    print(f"  {lab:12s} dome vs zero-wind outdoor={b['dome']:+.2f} (se {se['dome']:.2f}); wind={b['wind_o']:+.3f}/mph (se {se['wind_o']:.3f}); <20F={b['cold20']:+.2f} (se {se['cold20']:.2f}); "
          f"=> dome vs AVERAGE outdoor ({mw:.1f} mph) = {b['dome'] + b['wind_o']*mw:+.2f}")

print("\n== (d) OOS 2022-2025: LEAN with vs without dome term (fit 2009-2021) ==")
d9 = m[m.elo_sum.notna()].copy(); tr, te = d9[d9.train], d9[d9.test]
def fp(cols):
    X = sm.add_constant(tr[cols].astype(float)); f = sm.OLS(tr.y, X).fit()
    return f, te.lg_blend + f.predict(sm.add_constant(te[cols].astype(float)))
base_cols = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "wind_f", "div"]
f0, p0 = fp(base_cols); f1, p1 = fp(base_cols + ["dome"])
db, lo, hi, n = paired_mae_ci(p1 - te.total_pts, p0 - te.total_pts)
print(f"  dome coef in LEAN (with wind_f, so vs average outdoor conditions) = {f1.params['dome']:+.2f} (se {f1.bse['dome']:.2f}, p={f1.pvalues['dome']:.3f})")
print(f"  OOS MAE without dome={mae(p0, te.total_pts):.3f} with dome={mae(p1, te.total_pts):.3f}  dMAE={db:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")
sel = te.dome == 1
print(f"  dome games only (n={int(sel.sum())}): MAE without={mae(p0[sel], te.total_pts[sel]):.3f} with={mae(p1[sel], te.total_pts[sel]):.3f} market={mae(te.mkt_total[sel], te.total_pts[sel]):.3f}; bias without={(p0[sel]-te.total_pts[sel]).mean():+.2f} with={(p1[sel]-te.total_pts[sel]).mean():+.2f}")
# alternative: dome effect measured relative to GOOD outdoor weather, with wind in model: fit LEAN + dome on games where outdoor has observed weather
