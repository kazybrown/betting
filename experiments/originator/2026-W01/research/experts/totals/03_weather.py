"""THEORY 3: weather. Wind (10/15/20/25 mph), temperature (<32F, <20F), precipitation
(2023-25 only, nflfastR weather strings) on TOTALS and on SPREADS (favorites in wind).
Two different questions, two different baselines:
  (i)  what the MARKET misses: residual total_pts - mkt_total by weather bin (market total
       already knows the teams; if the residual is ~0 the market prices the weather).
  (ii) what an ORIGINATOR must subtract from a market-free number: residual of the LEAN
       team model WITHOUT weather (fit <=2021) by weather bin, i.e. the raw effect after
       controlling for the teams' scoring / QBs / Elo. This is what the spec table needs.
Weather sample = outdoor games with OBSERVED wind and temp (no imputation).
OOS check: does adding the fitted wind table to LEAN-no-weather lower 2022-25 MAE?
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/totals")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from common import build, mae, paired_mae_ci, ou_rate

K_TEAM, K_LG = 1, 128   # from 02 (validation-selected)
m = build(K_team=K_TEAM, K_lg=K_LG, verbose=False)
m = m[(m.game_type == "REG") & m.lg_prev.notna()].copy()
m["dome"] = m.is_dome.astype(int)
m["wx"] = ((m.outdoor == 1) & m.wind.notna() & m.temp.notna()).astype(int)
wbins = [-1, 9.99, 14.99, 19.99, 24.99, 200]; wlab = ["0-9", "10-14", "15-19", "20-24", "25+"]
m["wind_bin"] = pd.cut(m.wind, wbins, labels=wlab)
tbins = [-100, 19.99, 31.99, 49.99, 84.99, 200]; tlab = ["<20F", "20-31F", "32-49F", "50-84F", "85F+"]
m["temp_bin"] = pd.cut(m.temp, tbins, labels=tlab)


def summarize(df, col, bins, err, lab):
    print(f"\n  {lab}  (mean error, se, n, under-rate; error = actual - line)")
    for b in bins:
        x = df[df[col] == b]
        if len(x) == 0: continue
        e = x[err]
        print(f"    {str(b):8s} n={len(x):5d}  mean={e.mean():+6.2f}  se={e.std()/np.sqrt(len(x)):.2f}  under={(e<0).sum()/max((e!=0).sum(),1):.3f}")


# ---------------- (i) what the market misses ----------------
print("== (i) MARKET residual (total_pts - mkt_total) by weather bin, outdoor games with observed weather ==")
w = m[m.wx == 1]
for lab, d in [("ALL 1999-2025", w), ("FIT 1999-2021", w[w.train]), ("TEST 2022-2025", w[w.test])]:
    print(f"\n-- {lab}: n={len(d)} | domes for reference: n={int((m.dome==1).sum() if lab.startswith('ALL') else (m[m.train if 'FIT' in lab else m.test].dome==1).sum())}")
    summarize(d, "wind_bin", wlab, "total_err_mkt", "wind")
    summarize(d, "temp_bin", tlab, "total_err_mkt", "temperature")
dref = m[m.dome == 1]
print(f"\n  dome/closed reference: n={len(dref)} market residual mean={dref.total_err_mkt.mean():+.2f} se={dref.total_err_mkt.std()/np.sqrt(len(dref)):.2f}")
for lab, d in [("FIT 1999-2021", w[w.train]), ("TEST 2022-2025", w[w.test]), ("ALL", w)]:
    f = smf.ols("total_err_mkt ~ wind + I(temp < 32) + I(temp < 20)", data=d).fit(cov_type="HC1")
    print(f"  {lab:15s} market residual ~ wind + cold: wind={f.params['wind']:+.3f}/mph (p={f.pvalues['wind']:.3f}) "
          f"<32F={f.params['I(temp < 32)[T.True]']:+.2f} (p={f.pvalues['I(temp < 32)[T.True]']:.2f}) <20F={f.params['I(temp < 20)[T.True]']:+.2f} (p={f.pvalues['I(temp < 20)[T.True]']:.2f}) n={len(d)}")
for thr in (10, 15, 20, 25):
    a = w[w.wind >= thr]; b = w[w.wind < thr]
    t = stats.ttest_ind(a.total_err_mkt, b.total_err_mkt, equal_var=False)
    print(f"  wind >= {thr:2d}: n={len(a):4d} market residual {a.total_err_mkt.mean():+.2f} vs <{thr}: {b.total_err_mkt.mean():+.2f}  diff={a.total_err_mkt.mean()-b.total_err_mkt.mean():+.2f}  p={t.pvalue:.3f}  under-rate={((a.total_err_mkt<0).sum()/max((a.total_err_mkt!=0).sum(),1)):.3f}")

# ---------------- (ii) originator table: raw effect after controlling teams ----------------
print("\n== (ii) ORIGINATOR baseline: LEAN team model WITHOUT weather (fit 2009-2021, uses Elo+PF/PA+QB+dome+div), residual by bin ==")
d9 = m[m.elo_sum.notna()].copy()
BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "div"]
tr = d9[d9.train]
X = sm.add_constant(tr[BASE].astype(float)); f0 = sm.OLS(tr.total_pts - tr.lg_blend, X).fit()
d9["base_pred"] = d9.lg_blend + f0.predict(sm.add_constant(d9[BASE].astype(float)))
d9["base_err"] = d9.total_pts - d9.base_pred
w9 = d9[d9.wx == 1]
for lab, d in [("FIT 2009-2021 (in-sample for the team model)", w9[w9.train]), ("TEST 2022-2025", w9[w9.test]), ("ALL 2009-2025", w9)]:
    print(f"\n-- {lab}: n={len(d)}")
    summarize(d, "wind_bin", wlab, "base_err", "wind")
    summarize(d, "temp_bin", tlab, "base_err", "temperature")
# linear and binned fits with a "good weather" reference (wind<10, temp>=50)
w9 = w9.copy(); w9["w10"] = (w9.wind >= 10).astype(int); w9["w15"] = (w9.wind >= 15).astype(int); w9["w20"] = (w9.wind >= 20).astype(int); w9["w25"] = (w9.wind >= 25).astype(int)
w9["c32"] = (w9.temp < 32).astype(int); w9["c20"] = (w9.temp < 20).astype(int); w9["hot"] = (w9.temp >= 85).astype(int)
for lab, d in [("FIT 2009-2021", w9[w9.train]), ("TEST 2022-2025", w9[w9.test])]:
    f = smf.ols("base_err ~ wind + c32 + c20 + hot", data=d).fit(cov_type="HC1")
    print(f"  {lab}: linear  wind={f.params['wind']:+.3f}/mph (se {f.bse['wind']:.3f}, p={f.pvalues['wind']:.3f}) "
          f"<32F={f.params['c32']:+.2f} (p={f.pvalues['c32']:.2f}) <20F extra={f.params['c20']:+.2f} (p={f.pvalues['c20']:.2f}) 85F+={f.params['hot']:+.2f} (p={f.pvalues['hot']:.2f}) n={len(d)}")
    f = smf.ols("base_err ~ w10 + w15 + w20 + w25 + c32 + c20", data=d).fit(cov_type="HC1")
    cum = np.cumsum([f.params[k] for k in ["w10", "w15", "w20", "w25"]])
    print(f"  {lab}: stepped cumulative effect vs <10mph: 10-14={cum[0]:+.2f} 15-19={cum[1]:+.2f} 20-24={cum[2]:+.2f} 25+={cum[3]:+.2f} "
          f"(increments p: {f.pvalues['w10']:.2f} {f.pvalues['w15']:.2f} {f.pvalues['w20']:.2f} {f.pvalues['w25']:.2f})")
# fitted on all 1999-2021 outdoor games WITHOUT Elo (bigger n): pf/pa/dome baseline
print("\n  larger-n check 1999-2021 (no Elo/QB; baseline lg_blend + pf_sum + pa_sum + div):")
dl = m.copy(); trl = dl[dl.train]
Xl = sm.add_constant(trl[["pf_sum", "pa_sum", "div"]].astype(float)); fl = sm.OLS(trl.total_pts - trl.lg_blend, Xl).fit()
dl["base_err"] = dl.total_pts - dl.lg_blend - fl.predict(sm.add_constant(dl[["pf_sum", "pa_sum", "div"]].astype(float)))
wl = dl[(dl.wx == 1) & dl.train]
f = smf.ols("base_err ~ wind + I(temp < 32) + I(temp < 20)", data=wl).fit(cov_type="HC1")
print(f"    wind={f.params['wind']:+.3f}/mph (se {f.bse['wind']:.3f}) <32F={f.params['I(temp < 32)[T.True]']:+.2f} (se {f.bse['I(temp < 32)[T.True]']:.2f}) <20F extra={f.params['I(temp < 20)[T.True]']:+.2f} (se {f.bse['I(temp < 20)[T.True]']:.2f}) n={len(wl)}")
summarize(wl, "wind_bin", wlab, "base_err", "wind (1999-2021, team-controlled)")
# is the wind effect on the raw total linear or a threshold? residual by mph 0..30 smoothed
print("\n  team-controlled residual by wind mph (1999-2021 outdoor):")
for lo, hi in [(0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 25), (26, 60)]:
    x = wl[(wl.wind >= lo) & (wl.wind <= hi)]
    print(f"    {lo:2d}-{hi:2d} mph n={len(x):4d} mean={x.base_err.mean():+.2f} se={x.base_err.std()/np.sqrt(max(len(x),1)):.2f}")

# ---------------- OOS: does the wind table help an originator? ----------------
print("\n== OOS 2022-2025: LEAN-no-weather + wind adjustment (fit 2009-2021) ==")
te = d9[d9.test].copy()
mk_err = te.mkt_total - te.total_pts
trw = w9[w9.train]
f_lin = smf.ols("base_err ~ wind + c32", data=trw).fit()
te["wind_o"] = np.where(te.outdoor == 1, te.wind, np.nan)
te["adj_lin"] = np.where(te.wind_o.notna(), f_lin.params["wind"] * te.wind_o + f_lin.params["c32"] * (te.temp < 32), 0.0)
# center: the base model was fit including windy games, so the linear term must be centered at mean wind of outdoor train games
mean_wind = trw.wind.mean()
te["adj_lin_c"] = np.where(te.wind_o.notna(), f_lin.params["wind"] * (te.wind_o - mean_wind) + f_lin.params["c32"] * ((te.temp < 32).astype(float) - trw.c32.mean()), 0.0)
spec = lambda wnd: np.where(wnd >= 30, -4.0, np.where(wnd >= 21, -2.5, np.where(wnd >= 15, -1.0, 0.0)))
te["adj_spec"] = np.where(te.wind_o.notna(), spec(te.wind_o.fillna(0)) + np.where(te.temp < 20, -1.0, 0.0), 0.0)
fs = smf.ols("base_err ~ w15 + w20 + w25 + c32", data=trw).fit()
step = lambda wnd: np.where(wnd >= 25, fs.params.w15 + fs.params.w20 + fs.params.w25, np.where(wnd >= 20, fs.params.w15 + fs.params.w20, np.where(wnd >= 15, fs.params.w15, 0.0)))
te["adj_step"] = np.where(te.wind_o.notna(), step(te.wind_o.fillna(0)) + fs.params.c32 * (te.temp < 32), 0.0)
print(f"  fitted linear: {f_lin.params['wind']:+.3f}/mph, <32F {f_lin.params['c32']:+.2f}; fitted step: >=15 {fs.params.w15:+.2f}, >=20 extra {fs.params.w20:+.2f}, >=25 extra {fs.params.w25:+.2f}, <32F {fs.params.c32:+.2f}")
print(f"  (test games with observed outdoor weather: {int(te.wind_o.notna().sum())} of {len(te)}; the rest get adj=0)")
print(f"  {'variant':44s} MAE     bias   dMAE vs base [95% CI]      dMAE vs market [95% CI]     O/U vs mkt")
b0 = te.base_pred
for lab, p in [("base (no weather)", b0), ("base + spec table (15-20:-1, 21-30:-2.5, 30+:-4, <20F:-1)", b0 + te.adj_spec),
               ("base + fitted linear (uncentered)", b0 + te.adj_lin), ("base + fitted linear (centered at train mean wind)", b0 + te.adj_lin_c),
               ("base + fitted step table", b0 + te.adj_step)]:
    db, lo, hi, n = paired_mae_ci(p - te.total_pts, b0 - te.total_pts)
    dm, lo2, hi2, n = paired_mae_ci(p - te.total_pts, mk_err)
    wn, l, pu = ou_rate(p, te.mkt_total, te.total_pts)
    print(f"  {lab:44s} {mae(p, te.total_pts):.3f} {(p-te.total_pts).mean():+6.2f}  {db:+.3f} [{lo:+.3f},{hi:+.3f}]   {dm:+.3f} [{lo2:+.3f},{hi2:+.3f}]   {wn}-{l}-{pu}")
sel = te.wind_o >= 15
print(f"  restricted to test games with wind>=15 (n={int(sel.sum())}): base MAE={mae(b0[sel], te.total_pts[sel]):.3f}  +spec={mae((b0+te.adj_spec)[sel], te.total_pts[sel]):.3f}  +linear_c={mae((b0+te.adj_lin_c)[sel], te.total_pts[sel]):.3f}  +step={mae((b0+te.adj_step)[sel], te.total_pts[sel]):.3f}  market={mae(te.mkt_total[sel], te.total_pts[sel]):.3f}")

# ---------------- precipitation (2023-2025 only) ----------------
print("\n== precipitation proxy (nflfastR weather string, 2023-2025 outdoor games only) ==")
pz = m[(m.season >= 2023) & (m.outdoor == 1) & m.precip_any.notna()]
for col in ["precip_any", "precip_strict"]:
    a = pz[pz[col] == 1]; b = pz[pz[col] == 0]
    t = stats.ttest_ind(a.total_err_mkt, b.total_err_mkt, equal_var=False)
    print(f"  {col:14s} n={len(a):3d} vs dry n={len(b):4d}: market residual {a.total_err_mkt.mean():+.2f} (se {a.total_err_mkt.std()/np.sqrt(len(a)):.2f}) vs {b.total_err_mkt.mean():+.2f}; "
          f"diff={a.total_err_mkt.mean()-b.total_err_mkt.mean():+.2f} p={t.pvalue:.3f}; under-rate wet={(a.total_err_mkt<0).sum()/max((a.total_err_mkt!=0).sum(),1):.3f}; "
          f"raw total wet={a.total_pts.mean():.1f} dry={b.total_pts.mean():.1f}; mkt total wet={a.mkt_total.mean():.1f} dry={b.mkt_total.mean():.1f}")
pz9 = d9[(d9.season >= 2023) & (d9.outdoor == 1) & d9.precip_any.notna()]
a = pz9[pz9.precip_strict == 1]; b = pz9[pz9.precip_strict == 0]
print(f"  team-controlled (LEAN-no-weather residual): wet n={len(a)} {a.base_err.mean():+.2f} (se {a.base_err.std()/np.sqrt(len(a)):.2f}) vs dry {b.base_err.mean():+.2f}")

# ---------------- spreads: favorites in wind ----------------
print("\n== SPREADS: do favorites cover less in wind? (outdoor games with observed weather, 1999-2025) ==")
s = w.copy()
s = s[s.mkt_spread.notna() & (s.mkt_spread != 0)]
s["fav_home"] = (s.mkt_spread < 0).astype(int)
s["abs_spread"] = s.mkt_spread.abs()
# favorite cover margin: favorite's actual margin minus the number it laid
s["fav_cover_margin"] = np.where(s.fav_home == 1, s.margin - s.abs_spread, -s.margin - s.abs_spread)
s["fav_cover"] = (s.fav_cover_margin > 0).astype(int); s["push"] = (s.fav_cover_margin == 0)
for lab, d in [("ALL", s), ("FIT <=2021", s[s.train]), ("TEST 2022-25", s[s.test])]:
    print(f"-- {lab} n={len(d)}")
    for b in wlab:
        x = d[d.wind_bin == b]; xx = x[~x.push]
        print(f"    wind {b:6s} n={len(x):5d} fav cover margin={x.fav_cover_margin.mean():+.2f} (se {x.fav_cover_margin.std()/np.sqrt(max(len(x),1)):.2f})  fav ATS={xx.fav_cover.mean():.3f}  spread MAE={x.spread_err_mkt.abs().mean():.2f}  mean |spread|={x.abs_spread.mean():.1f}  fav scored share={((np.where(x.fav_home==1, x.home_score, x.away_score))/x.total_pts.replace(0,np.nan)).mean():.3f}")
    f = smf.ols("fav_cover_margin ~ wind + wind:abs_spread + abs_spread", data=d).fit(cov_type="HC1")
    print(f"    OLS fav_cover_margin ~ wind*|spread|: wind={f.params['wind']:+.3f} (p={f.pvalues['wind']:.2f}) wind:|spread|={f.params['wind:abs_spread']:+.4f} (p={f.pvalues['wind:abs_spread']:.2f}) |spread|={f.params['abs_spread']:+.3f} (p={f.pvalues['abs_spread']:.2f})")
    for thr in (15, 20):
        a = d[d.wind >= thr]; b = d[d.wind < thr]
        t = stats.ttest_ind(a.fav_cover_margin, b.fav_cover_margin, equal_var=False)
        print(f"    wind>={thr}: n={len(a)} fav cover margin {a.fav_cover_margin.mean():+.2f} vs {b.fav_cover_margin.mean():+.2f} (p={t.pvalue:.3f}); fav ATS {a[~a.push].fav_cover.mean():.3f} vs {b[~b.push].fav_cover.mean():.3f}; "
              f"big favs (|spread|>=7) in wind: n={int(((a.abs_spread>=7)).sum())} ATS={a[(a.abs_spread>=7)&~a.push].fav_cover.mean():.3f}")
# does wind shrink the spread error scale (so should the SD/confidence change)?
print("\n  spread MAE / margin SD by wind (all outdoor):")
for b in wlab:
    x = s[s.wind_bin == b]
    print(f"    wind {b:6s} n={len(x):5d} spread MAE={x.spread_err_mkt.abs().mean():.2f} margin SD={x.margin.std():.2f} total SD={x.total_pts.std():.2f} total MAE(mkt)={x.total_err_mkt.abs().mean():.2f}")
