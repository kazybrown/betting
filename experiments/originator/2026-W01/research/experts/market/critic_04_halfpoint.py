"""critic_04_halfpoint.py - adversarial re-analysis of T4 (half-point value table / probability-space edges).
Attacks: (a) reproduce masses; 3-block stability with Wilson CIs; far-side conditioning (mass at 3 when the
line is 4.5-6 vs 1-2.5); (b) the formula assumes the ORIGIN is the truth - estimate the shrinkage factor
lambda of origin-vs-market gaps with an honest rolling-origin Elo+HFA+QB origin (fit on seasons < s) and with
nfelo's close; (c) the direct empirical test of T4's premise: does 'crossed mass' predict realized covers
BEYOND raw point gap? logistic cover ~ gap + crossed_mass, plus cross-3/7 vs no-key tables at equal gaps;
(d) realized ROI at ACTUAL closing prices of the recommended rule 'act when formula EV >= 0.02' by EV bucket;
(e) the -3 / -7 guard: dog +3 cover by block with juice-implied probability.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_04_halfpoint.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae
def wil(k, n):
    r = stats.binomtest(int(k), int(n), 0.5); c = r.proportion_ci(method="wilson"); return r.pvalue, c.low, c.high
def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.maximum(np.abs(o), 100))

m = merged(); m = m[m.mkt_spread.notna()].copy()
m["fav_line"] = m.mkt_spread.abs(); m["fav_margin"] = np.where(m.mkt_spread < 0, m.margin, np.where(m.mkt_spread > 0, -m.margin, m.margin))
m["block"] = pd.cut(m.season, [2008, 2014, 2021, 2025], labels=["2009-14", "2015-21", "2022-25"])

print("=== (a) landing masses: reproduce, blocks, far-side conditioning ===")
def mass(df, k, w=2):
    g = df[(df.fav_line >= k - w) & (df.fav_line <= k + w)]; c = int((g.fav_margin == k).sum()); return c / len(g), len(g), c
rows = []
for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 14]:
    p, n, c = mass(m, k); _, lo, hi = wil(c, n); r = dict(k=k, n=n, mass=p, ci_lo=lo, ci_hi=hi)
    for blk, g in m.groupby("block", observed=True):
        pb, nb, cb = mass(g, k); r[f"m_{blk}"] = pb
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
for k in [3, 7]:
    for lo, hi in [(0.5, k - 1.5), (k - 1, k - 0.5), (k, k + 0.5), (k + 1, k + 2), (k + 2.5, k + 4)]:
        g = m[(m.fav_line >= lo) & (m.fav_line <= hi)]; c = int((g.fav_margin == k).sum()); _, l_, h_ = wil(c, len(g))
        print(f"  mass at {k} | line in [{lo},{hi}]: {c/len(g):.3f} (CI {l_:.3f}-{h_:.3f}, n={len(g)})")

# ---------- honest rolling-origin Elo origin ----------
d = m[m.elo_dif_pts.notna()].copy(); d["qb_adj_pts"] = (d.home_538_qb_adj.fillna(0) - d.away_538_qb_adj.fillna(0)) / 25.0
cols = ["elo_dif_pts", "hfa_pts", "qb_adj_pts"]; out = []
for s in range(2012, 2026):
    fit = d[d.season < s]; te = d[d.season == s].copy()
    ols = sm.OLS(fit.margin, sm.add_constant(fit[cols])).fit(); te["origin"] = -ols.predict(sm.add_constant(te[cols])); out.append(te)
r = pd.concat(out)
print(f"\nrolling-origin Elo+HFA+QB origin (fit < season, 2012-25): n={len(r)} | MAE origin {mae(-r.origin, r.margin):.3f} vs market {mae(-r.mkt_spread, r.margin):.3f} | SD(origin - market) {(r.origin - r.mkt_spread).std():.2f}")
print("\n=== (b) shrinkage lambda: residual-vs-market ~ (market - origin)  [lambda=1: origin is truth; 0: gap is pure noise] ===")
for name, g, col in [("Elo origin OOS 2012-25", r, "origin"), ("nfelo close 2009-25", m[m.nfelo_home_line_close.notna()], "nfelo_home_line_close")]:
    gg = g.copy(); gg["gapv"] = gg.mkt_spread - gg[col]; gg["res"] = gg.margin + gg.mkt_spread
    o = sm.OLS(gg.res, sm.add_constant(gg.gapv)).fit(cov_type="HC1")
    print(f"{name}: n={len(gg)} lambda={o.params['gapv']:+.3f} (SE {o.bse['gapv']:.3f}, p={o.pvalues['gapv']:.3f}) | SD gap {gg.gapv.std():.2f}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 99)]:
        h = gg[(gg.gapv.abs() >= lo) & (gg.gapv.abs() < hi) & (gg.gapv != 0)]; s_ = h[h.res != 0]; k = int(((np.sign(s_.res) == np.sign(s_.gapv))).sum())
        if len(s_) > 30: print(f"   |gap| {lo}-{hi}: n={len(s_)} origin-side cover {k/len(s_):.3f} (CI {wil(k, len(s_))[1]:.3f}-{wil(k, len(s_))[2]:.3f})")

# ---------- (c) crossed mass beyond gap ----------
fitp = m[m.season <= 2021]
MASS = {k: mass(fitp, k)[0] for k in range(1, 25)}
def build(g, col):
    g = g.copy(); sgn = np.where(g[col] < 0, 1, -1)              # +1: origin favors home
    g["x"] = g[col].abs(); g["mk"] = -sgn * g.mkt_spread; g["fm"] = sgn * g.margin
    g["gapv"] = g.mk - g.x                                        # >0: take origin's dog; <0: lay origin's fav
    g["side"] = np.where(g.gapv > 0, -1, 1)
    def crossed(x, mk):
        lo, hi = min(x, mk), max(x, mk); ks = [k for k in range(1, 25) if lo < k < hi]; return sum(MASS[k] for k in ks), int(3 in ks), int(7 in ks), len(ks)
    cm = np.array([crossed(x, mk) for x, mk in zip(g.x, g.mk)]); g["cmass"] = cm[:, 0]; g["c3"] = cm[:, 1]; g["c7"] = cm[:, 2]; g["nint"] = cm[:, 3]
    g["push_m"] = np.where(g.mk % 1 == 0, [MASS.get(int(abs(v)), 0.0) for v in g.mk], 0.0)
    pw = 0.5 + g.cmass; pl = np.clip(1 - pw - g.push_m, 0, None); g["ev_formula"] = (pw - 1.1 * pl) / 1.1
    g["win"] = np.where(g.fm == g.mk, np.nan, ((g.fm > g.mk) == (g.side > 0)).astype(float))
    fav_is_home = sgn > 0
    side_home = np.where(g.side > 0, fav_is_home, ~fav_is_home)
    g["odds"] = np.where(side_home, g.home_spread_odds, g.away_spread_odds)
    g["ret"] = np.where(g.win.isna(), 0.0, np.where(g.win == 1, dec(g.odds) - 1, -1.0))
    g["ret110"] = np.where(g.win.isna(), 0.0, np.where(g.win == 1, 1 / 1.1, -1.0))
    return g
for name, g, col in [("Elo origin OOS 2012-25", r, "origin"), ("nfelo close 2009-25", m[m.nfelo_home_line_close.notna()], "nfelo_home_line_close")]:
    b = build(g, col); b = b[b.gapv.abs() >= 0.5]
    print(f"\n=== (c) {name}: bets at |gap|>=0.5, n={len(b)} ===")
    bb = b.dropna(subset=["win"])
    X = sm.add_constant(pd.DataFrame({"gap": bb.gapv.abs(), "cmass": bb.cmass}))
    lg = sm.Logit(bb.win, X).fit(disp=0)
    print(f"logit cover ~ |gap| + crossed_mass: b_gap={lg.params['gap']:+.3f} (p={lg.pvalues['gap']:.3f}) b_cmass={lg.params['cmass']:+.2f} (p={lg.pvalues['cmass']:.3f}) | T4 premise needs b_cmass>0 beyond gap")
    X2 = sm.add_constant(pd.DataFrame({"gap": bb.gapv.abs(), "c3": bb.c3, "c7": bb.c7}))
    lg2 = sm.Logit(bb.win, X2).fit(disp=0)
    print(f"logit cover ~ |gap| + cross3 + cross7: b_gap={lg2.params['gap']:+.3f} (p={lg2.pvalues['gap']:.3f}) b_c3={lg2.params['c3']:+.3f} (p={lg2.pvalues['c3']:.3f}) b_c7={lg2.params['c7']:+.3f} (p={lg2.pvalues['c7']:.3f})")
    rows = []
    for lo, hi in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 99)]:
        for lab, mk in [("crosses 3 or 7", (b.c3 + b.c7) > 0), ("no 3/7 crossed", (b.c3 + b.c7) == 0)]:
            h = b[(b.gapv.abs() >= lo) & (b.gapv.abs() < hi) & mk]; s_ = h.dropna(subset=["win"]); k = int(s_.win.sum())
            if len(s_) < 20: continue
            p, l_, h_ = wil(k, len(s_))
            rows.append(dict(gap=f"{lo}-{hi}", key=lab, n=len(h), cover=k / len(s_), ci_lo=l_, ci_hi=h_, roi_actual=h.ret.mean(), roi_110=h.ret110.mean(), mean_formula_ev=h.ev_formula.mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"--- (d) {name}: by formula-EV bucket (the recommended rule acts at EV>=0.02) ---")
    rows = []
    for lo, hi in [(-1, 0.0), (0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 9)]:
        h = b[(b.ev_formula >= lo) & (b.ev_formula < hi)]; s_ = h.dropna(subset=["win"]); k = int(s_.win.sum())
        if len(s_) < 20: continue
        p, l_, h_ = wil(k, len(s_)); rows.append(dict(ev_bucket=f"{lo}..{hi}", n=len(h), share_of_games=len(h) / len(g), cover=k / len(s_), ci_lo=l_, ci_hi=h_, roi_actual=h.ret.mean(), roi_110=h.ret110.mean(), units_actual=h.ret.sum()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    h = b[b.ev_formula >= 0.02]; s_ = h.dropna(subset=["win"]); k = int(s_.win.sum())
    print(f"rule 'act when EV>=0.02': bets {len(h)} = {len(h)/len(g):.1%} of games | cover {k/len(s_):.3f} (CI {wil(k,len(s_))[1]:.3f}-{wil(k,len(s_))[2]:.3f}) | ROI actual {h.ret.mean():+.3f} | units {h.ret.sum():+.1f}")
    # realized value of crossing 3 at a fixed gap band, difference test
    a = b[(b.gapv.abs() >= 0.5) & (b.gapv.abs() < 1.5) & (b.c3 == 1)].dropna(subset=["win"]); c = b[(b.gapv.abs() >= 0.5) & (b.gapv.abs() < 1.5) & (b.c3 == 0) & (b.c7 == 0)].dropna(subset=["win"])
    if len(a) > 20 and len(c) > 20:
        p1, p2 = a.win.mean(), c.win.mean(); pp = (a.win.sum() + c.win.sum()) / (len(a) + len(c)); z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(c)))
        print(f"gap 0.5-1.5 crossing 3 (n={len(a)}) cover {p1:.3f} vs no key (n={len(c)}) {p2:.3f}: diff {p1-p2:+.3f}, z={z:+.2f}, p={2*(1-stats.norm.cdf(abs(z))):.3f} | formula predicts diff ~ +{MASS[3]:.3f} x lambda")

print("\n=== (e) the -3/-7 guard: dogs at exactly +3 / +7 by block, with juice-implied dog prob ===")
j = m[m.home_spread_odds.notna()].copy(); j["fav_odds"] = np.where(j.mkt_spread < 0, j.home_spread_odds, j.away_spread_odds); j["dog_odds"] = np.where(j.mkt_spread < 0, j.away_spread_odds, j.home_spread_odds)
j["p_dog_imp"] = (1 / dec(j.dog_odds)) / (1 / dec(j.dog_odds) + 1 / dec(j.fav_odds))
for k in [3, 7]:
    for blk, g in list(j[j.fav_line == k].groupby("block", observed=True)) + [("ALL", j[j.fav_line == k])]:
        dec_ = g[g.fav_margin != k]; kk = int((dec_.fav_margin < k).sum()); p, lo, hi = wil(kk, len(dec_))
        ret = np.where(dec_.fav_margin < k, dec(dec_.dog_odds) - 1, -1.0)
        print(f"  +{k} dogs {blk}: n={len(g)} push {(g.fav_margin == k).mean():.3f} | dog wins {kk}/{len(dec_)} = {kk/len(dec_):.3f} (CI {lo:.3f}-{hi:.3f}, p={p:.3f}) | juice-implied dog {g.p_dog_imp.mean():.3f} | dog ROI at actual price {ret.mean():+.3f}")
    # the guard cell: latent k+0.25 laying k -> pool of games closed at k and k+0.5
    for blk, g in list(m.groupby("block", observed=True)) + [("ALL", m)]:
        s_ = g[g.fav_line.isin([k, k + 0.5])].fav_margin.values; pw, pp, pl = (s_ > k).mean(), (s_ == k).mean(), (s_ < k).mean()
        print(f"  guard cell latent {k+0.25} lay {k} {blk}: n={len(s_)} fav EV(-110) {(pw - 1.1*pl)/1.1:+.3f} [P(win) {pw:.3f} push {pp:.3f}]")
