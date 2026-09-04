"""CRITIC 04 - T4 dome effect.
Reproduce the participant team-season FE estimate, then: (1) placebo pseudo-domes (random outdoor home teams
flagged, same FE design, outdoor games only) - is the design biased toward positive 'home venue' effects?
(2) leave-one-dome-team-out; (3) within-stadium closed vs open at retractable roofs; (4) market residual by
roof with season clustering and by era; (5) marginal dome coefficient in the V3 spec by rolling origin.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from critic_common import build_fixed, mae, paired_mae_ci

m = build_fixed(K_team=1, K_lg=128)
m = m[(m.game_type == "REG") & m.lg_prev.notna() & ~m.neutral].copy()
m["y"] = m.total_pts - m.lg_blend
m["hts"] = m.home + "_" + m.season.astype(str); m["ats"] = m.away + "_" + m.season.astype(str)
m["stad"] = m.stadium.fillna("unk").str.strip()


def fe_fit(d, xcols, cluster=None):
    ts = sorted(set(d.hts) | set(d.ats)); idx = {t: i for i, t in enumerate(ts)}
    F = np.zeros((len(d), len(ts)))
    F[np.arange(len(d)), d.hts.map(idx).values] += 1.0; F[np.arange(len(d)), d.ats.map(idx).values] += 1.0
    X = np.hstack([d[xcols].astype(float).values, F])
    if cluster is not None:
        r = sm.OLS(d.y.values.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    else:
        r = sm.OLS(d.y.values.astype(float), X).fit(cov_type="HC1")
    k = len(xcols); return dict(zip(xcols, r.params[:k])), dict(zip(xcols, r.bse[:k]))


print("== reproduce: dome/closed vs outdoors, participant team-season FE ==")
for lab, d in [("ALL 1999-2025", m), ("FIT <=2021", m[m.train]), ("TEST 2022-25", m[m.test])]:
    b, se = fe_fit(d, ["dome"]); b2, se2 = fe_fit(d, ["dome"], cluster="stad")
    print(f"  {lab:14s} n={len(d)} dome={b['dome']:+.2f} (HC1 se {se['dome']:.2f}; stadium-cluster se {se2['dome']:.2f})")

print("\n== (1) placebo: outdoor games only; flag home games of k random outdoor home teams per season (k = # dome teams that season); 200 draws ==")
o = m[m.dome == 0].copy()
n_dome_teams = m[m.dome == 1].groupby("season").home.nunique()
rng = np.random.default_rng(7); bs = []
ts = sorted(set(o.hts) | set(o.ats)); idx = {t: i for i, t in enumerate(ts)}
F = np.zeros((len(o), len(ts))); F[np.arange(len(o)), o.hts.map(idx).values] += 1.0; F[np.arange(len(o)), o.ats.map(idx).values] += 1.0
Q, _ = np.linalg.qr(F)                       # residualize on the FE once (Frisch-Waugh)
y_r = o.y.values - Q @ (Q.T @ o.y.values)
for i in range(200):
    flag = np.zeros(len(o))
    for s, k in n_dome_teams.items():
        teams = o.loc[o.season == s, "home"].unique(); pick = rng.choice(teams, size=min(int(k), len(teams)), replace=False)
        flag[(o.season == s).values & o.home.isin(pick).values] = 1.0
    x_r = flag - Q @ (Q.T @ flag); bs.append(float((x_r @ y_r) / (x_r @ x_r)))
bs = np.array(bs)
print(f"  placebo pseudo-dome coefficient: mean={bs.mean():+.3f} sd={bs.std():.3f} 2.5/97.5%=[{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]  (real dome = +2.15)")
print(f"  share of placebo draws >= +2.15: {(bs >= 2.15).mean():.3f}")

print("\n== (2) leave-one-dome-home-team-out (ALL 1999-2025) ==")
dome_homes = m[m.dome == 1].groupby("home").size().sort_values(ascending=False)
print("  dome home teams (games):", dome_homes.to_dict())
for t in dome_homes.index:
    d = m[~((m.dome == 1) & (m.home == t))]
    b, se = fe_fit(d, ["dome"]); print(f"   drop {t:4s}: dome={b['dome']:+.2f} (se {se['dome']:.2f})")
print("  by dome home team (that team's dome games flagged alone, others as outdoor-equivalent excluded):")
for t in dome_homes.index[:8]:
    d = m[(m.dome == 0) | (m.home == t)].copy()
    b, se = fe_fit(d, ["dome"]); print(f"   {t:4s} n_dome={int((d.dome==1).sum())}: {b['dome']:+.2f} (se {se['dome']:.2f})")

print("\n== (3) within-stadium: closed vs open at retractable-roof stadiums (stadium x season FE + participant FE) ==")
retr = m[m.roof.isin(["closed", "open"])].copy()
retr_stads = retr.groupby("stad").roof.nunique(); both = retr_stads[retr_stads == 2].index
r = m[m.stad.isin(both) & m.roof.isin(["closed", "open"])].copy()
r["closed"] = (r.roof == "closed").astype(int)
r["ss"] = r.stad + "_" + r.season.astype(str)
ts = sorted(set(r.hts) | set(r.ats)); idx = {t: i for i, t in enumerate(ts)}
F = np.zeros((len(r), len(ts))); F[np.arange(len(r)), r.hts.map(idx).values] += 1.0; F[np.arange(len(r)), r.ats.map(idx).values] += 1.0
S = pd.get_dummies(r.ss).values.astype(float)
X = np.hstack([r[["closed"]].values.astype(float), F, S]); f = sm.OLS(r.y.values, X).fit(cov_type="HC1")
print(f"  stadiums with both: {list(both)}; n={len(r)} (closed {int(r.closed.sum())}, open {int((1-r.closed).sum())}); closed-vs-open within stadium-season = {f.params[0]:+.2f} (se {f.bse[0]:.2f})")
X = np.hstack([r[["closed"]].values.astype(float), S]); f = sm.OLS(r.y.values, X).fit(cov_type="HC1")
print(f"  stadium-season FE only (no participant FE): {f.params[0]:+.2f} (se {f.bse[0]:.2f}); raw mean total closed={r[r.closed==1].total_pts.mean():.1f} open={r[r.closed==0].total_pts.mean():.1f}")

print("\n== (4) market residual by roof: cluster by season; by era ==")
for lab, d in [("ALL", m), ("1999-2010", m[m.season <= 2010]), ("2011-2021", m[m.season.between(2011, 2021)]), ("2022-25", m[m.test])]:
    X = sm.add_constant(d[["dome"]].astype(float)); f = sm.OLS(d.total_err_mkt, X).fit(cov_type="cluster", cov_kwds={"groups": d.season})
    print(f"  {lab:10s} dome-outdoor market residual diff={f.params['dome']:+.2f} (season-cluster se {f.bse['dome']:.2f}, p={f.pvalues['dome']:.3f}) n_dome={int(d.dome.sum())}")

print("\n== (5) marginal dome coefficient inside the V3 spec (with PF/PA, QB, wind_c), rolling origin ==")
d9 = m[m.elo_sum.notna()].copy(); FULL = ["elo_sum", "pf_dev", "pa_dev", "qb_sum", "div", "dome", "wind_c", "cold20"]
for Y in (2016, 2019, 2022, 2026):
    a = d9[d9.season < Y]; X = sm.add_constant(a[FULL].astype(float)); f = sm.OLS(a.y, X).fit(cov_type="HC1")
    print(f"  fit < {Y}: dome={f.params['dome']:+.2f} (se {f.bse['dome']:.2f}) wind_c={f.params['wind_c']:+.3f} n={len(a)}")
a = d9[d9.test]; X = sm.add_constant(a[FULL].astype(float)); f = sm.OLS(a.y, X).fit(cov_type="HC1")
print(f"  test 2022-25 in-sample: dome={f.params['dome']:+.2f} (se {f.bse['dome']:.2f}) n={len(a)}")
