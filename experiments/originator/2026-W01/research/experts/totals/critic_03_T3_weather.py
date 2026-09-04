"""CRITIC 03 - T3a wind, T3b temp/precip, T3c wind on spreads.
Team-controlled residual = total - [lg_blend + pf_sum + pa_sum + div fit <=2021] (expert's larger-n baseline),
outdoor REG games with observed wind/temp.
Attacks: (1) venue confound -> STADIUM fixed effects (wind identified within stadium) + cluster SE by stadium;
(2) era stability; (3) Huber / drop 25+ (are 38 games driving the slope?); (4) permutation placebo
(shuffle wind within season x stadium); (5) market residual with stadium FE; (6) bins with stadium FE;
(7) temperature with stadium FE; (8) spreads: margin ~ spread x wind, favourite cover margin with stadium FE, by era.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from critic_common import build_fixed, mae

m = build_fixed(K_team=1, K_lg=128)
m = m[(m.game_type == "REG") & m.lg_prev.notna()].copy()
tr = m[m.train]
X = sm.add_constant(tr[["pf_sum", "pa_sum", "div"]].astype(float)); fl = sm.OLS(tr.total_pts - tr.lg_blend, X).fit()
m["base_err"] = m.total_pts - m.lg_blend - fl.predict(sm.add_constant(m[["pf_sum", "pa_sum", "div"]].astype(float)))
w = m[(m.outdoor == 1) & m.wind.notna() & m.temp.notna()].copy()
w["stad"] = w.stadium.fillna("unk").str.strip()
w["c32"] = (w.temp < 32).astype(int); w["c20"] = (w.temp < 20).astype(int); w["hot"] = (w.temp >= 85).astype(int)
w["w25"] = (w.wind >= 25).astype(int)
print(f"outdoor REG games with observed wind/temp: n={len(w)} (fit<=2021 n={int(w.train.sum())}, test n={int(w.test.sum())}); stadiums={w.stad.nunique()}")

def slope(df, formula, cluster=None, robust=False):
    if robust:
        f = smf.rlm(formula, data=df, M=sm.robust.norms.HuberT()).fit(); return f.params["wind"], f.bse["wind"], np.nan
    if cluster:
        f = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    else:
        f = smf.ols(formula, data=df).fit(cov_type="HC1")
    return f.params["wind"], f.bse["wind"], f.pvalues["wind"]

print("\n== (1) team-controlled wind slope: pooled vs STADIUM fixed effects (within-stadium identification) ==")
for lab, df in [("FIT 1999-2021", w[w.train]), ("TEST 2022-25", w[w.test]), ("ALL", w)]:
    b1, s1, p1 = slope(df, "base_err ~ wind + c32 + c20")
    b2, s2, p2 = slope(df, "base_err ~ wind + c32 + c20", cluster="stad")
    b3, s3, p3 = slope(df, "base_err ~ wind + c32 + c20 + C(stad)", cluster="stad")
    b4, s4, p4 = slope(df, "base_err ~ wind + c32 + c20 + C(stad) + C(season)", cluster="stad")
    b5, s5, _ = slope(df, "base_err ~ wind + c32 + c20 + C(stad)", robust=True)
    b6, s6, p6 = slope(df[df.wind < 25], "base_err ~ wind + c32 + c20 + C(stad)", cluster="stad")
    print(f"  {lab:14s} n={len(df):4d} pooled={b1:+.3f}({s1:.3f}) | cluster-stad se={s2:.3f} | +stadium FE={b3:+.3f}({s3:.3f}, p={p3:.3f}) | +stad+season FE={b4:+.3f}({s4:.3f}) | Huber+stadFE={b5:+.3f}({s5:.3f}) | stadFE excl 25+={b6:+.3f}({s6:.3f}, p={p6:.3f})")

print("\n== (2) era stability of the team-controlled slope (stadium FE, cluster by stadium) ==")
for lo, hi in [(1999, 2004), (2005, 2010), (2011, 2016), (2017, 2021), (2022, 2025)]:
    df = w[w.season.between(lo, hi)]
    b, s, p = slope(df, "base_err ~ wind + c32 + c20 + C(stad)", cluster="stad")
    bm, sm_, pm = slope(df, "total_err_mkt ~ wind + c32 + c20 + C(stad)", cluster="stad")
    print(f"  {lo}-{hi}: n={len(df):4d} team-controlled={b:+.3f} (se {s:.3f}, p={p:.3f}) | market residual={bm:+.3f} (se {sm_:.3f}, p={pm:.3f}) | mean wind={df.wind.mean():.1f}")

print("\n== (3) permutation placebo: shuffle wind within (season, stadium), 300 draws, FIT sample, stadium-FE spec ==")
rng = np.random.default_rng(1); df = w[w.train].copy(); real, _, _ = slope(df, "base_err ~ wind + c32 + c20 + C(stad)")
bs = []
for i in range(300):
    df["wind_p"] = df.groupby(["season", "stad"]).wind.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    f = smf.ols("base_err ~ wind_p + c32 + c20 + C(stad)", data=df).fit(); bs.append(f.params["wind_p"])
bs = np.array(bs)
print(f"  real slope={real:+.3f}; placebo mean={bs.mean():+.4f} sd={bs.std():.4f}; share of |placebo| >= |real| = {(np.abs(bs) >= abs(real)).mean():.3f}")

print("\n== (4) bins with stadium FE (vs <10 mph), team-controlled, FIT 1999-2021 ==")
df = w[w.train].copy()
df["wb"] = pd.cut(df.wind, [-1, 9.99, 14.99, 19.99, 24.99, 200], labels=["0-9", "10-14", "15-19", "20-24", "25+"])
f = smf.ols("base_err ~ C(wb) + c32 + c20 + C(stad)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.stad})
for k in ["10-14", "15-19", "20-24", "25+"]:
    key = f"C(wb)[T.{k}]"; print(f"  {k:6s} {f.params[key]:+.2f} (se {f.bse[key]:.2f}) n={int((df.wb == k).sum())}")
f = smf.ols("total_err_mkt ~ C(wb) + c32 + c20 + C(stad)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.stad})
print("  market residual bins with stadium FE: " + "  ".join(f"{k}={f.params[f'C(wb)[T.{k}]']:+.2f}({f.bse[f'C(wb)[T.{k}]']:.2f})" for k in ["10-14", "15-19", "20-24", "25+"]))
# low-wind fine bins with stadium FE: is 'start at 10' right, or is 7-9 already down?
df["wb2"] = pd.cut(df.wind, [-1, 3, 6, 9, 12, 15, 18, 21, 25, 200], labels=["0-3", "4-6", "7-9", "10-12", "13-15", "16-18", "19-21", "22-25", "26+"])
f = smf.ols("base_err ~ C(wb2, Treatment('4-6')) + c32 + c20 + C(stad)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.stad})
KEY = "C(wb2, Treatment('4-6'))[T.{}]"
print("  fine bins vs 4-6 mph (stadium FE): " + "  ".join(f"{k}={f.params[KEY.format(k)]:+.2f}({f.bse[KEY.format(k)]:.2f})" for k in ["0-3", "7-9", "10-12", "13-15", "16-18", "19-21", "22-25", "26+"]))

print("\n== (5) temperature (T3b) with wind linear + stadium FE, team-controlled ==")
for lab, df in [("FIT 1999-2021", w[w.train]), ("ALL 1999-2025", w)]:
    f = smf.ols("base_err ~ wind + c32 + c20 + hot + C(stad)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.stad})
    print(f"  {lab}: <32F={f.params['c32']:+.2f} (se {f.bse['c32']:.2f}) <20F extra={f.params['c20']:+.2f} (se {f.bse['c20']:.2f}, n={int(df.c20.sum())}) 85F+={f.params['hot']:+.2f} (se {f.bse['hot']:.2f})")
print("  precipitation: flag exists only 2023-25 (= TEST). The expert's -1.5 term was sized on test data; n=39. Nothing to add.")

print("\n== (6) T3c spreads: does wind change the spread's predictive slope? margin ~ spread_line * wind (outdoor, observed wind) ==")
s = w[w.mkt_spread.notna()].copy(); s["sl"] = -s.mkt_spread   # sl = expected home margin (nflverse convention)
for lab, df in [("FIT <=2021", s[s.train]), ("TEST 2022-25", s[s.test]), ("ALL", s)]:
    f = smf.ols("margin ~ sl + sl:wind + wind", data=df).fit(cov_type="HC1")
    print(f"  {lab:12s} n={len(df)} slope on spread={f.params['sl']:+.3f} ; sl:wind={f.params['sl:wind']:+.4f} (se {f.bse['sl:wind']:.4f}, p={f.pvalues['sl:wind']:.2f}) -> slope at 20 mph = {f.params['sl'] + 20*f.params['sl:wind']:+.3f}")
s = s[s.mkt_spread != 0].copy(); s["fav_home"] = (s.mkt_spread < 0).astype(int); s["abs_spread"] = s.mkt_spread.abs()
s["fcm"] = np.where(s.fav_home == 1, s.margin - s.abs_spread, -s.margin - s.abs_spread)
for lab, df in [("FIT <=2021", s[s.train]), ("TEST 2022-25", s[s.test]), ("ALL", s)]:
    f = smf.ols("fcm ~ wind + C(stad)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.stad})
    a = df[df.wind >= 15]; b = df[df.wind < 15]
    print(f"  {lab:12s} fav cover margin ~ wind (stadium FE): {f.params['wind']:+.3f}/mph (se {f.bse['wind']:.3f}, p={f.pvalues['wind']:.2f}) | >=15: {a.fcm.mean():+.2f} (n={len(a)}) vs {b.fcm.mean():+.2f}")
