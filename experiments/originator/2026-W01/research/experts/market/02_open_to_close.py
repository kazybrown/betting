"""02_open_to_close.py - Theory 2: does open->close movement predict results BEYOND the close?
Data: nfelo_games.csv home_line_open / home_line_close (both NEGATIVE = home favored; same source
so the move is internally consistent). move = close - open; move<0 = line moved TOWARD home
("steam on home"). Residual vs close = margin + close (>0 home beat the close).
Tests: (a) MAE open vs close (how much info the move carries vs the OPEN);
       (b) OLS residual_vs_close ~ move, fit<=2021 / test 2022-25;
       (c) ATS of 'bet with steam at the close' by |move| bucket and era;
       (d) does steam agreeing with a model pick (nfelo close vs market close) improve that pick?
       (e) totals: only 2024-25 have total_line_open (n~570) -> reported but INCONCLUSIVE by design.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/02_open_to_close.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae, ats

m = merged()
d = m[m.home_line_open.notna() & m.home_line_close.notna() & m.margin.notna()].copy()
d["move"] = d.home_line_close - d.home_line_open       # <0: moved toward home
d["res_close"] = d.margin + d.home_line_close           # >0: home beat the close
d["res_open"] = d.margin + d.home_line_open
d["era"] = np.where(d.season <= 2021, "fit<=2021", "test2022-25")
print("n games with open & close:", len(d), "| seasons", d.season.min(), "-", d.season.max())
print("sign check corr(home_line_close, margin) =", round(np.corrcoef(d.home_line_close, d.margin)[0, 1], 3), "(negative expected)")
print("\nmove distribution (close - open):")
print(d.move.describe().round(3).to_string())
print("share no move:", round((d.move == 0).mean(), 3), "| |move|>=1:", round((d.move.abs() >= 1).mean(), 3), "| |move|>=2:", round((d.move.abs() >= 2).mean(), 3))

print("\n(a) MAE vs margin: open", round(mae(-d.home_line_open, d.margin), 3), "| close", round(mae(-d.home_line_close, d.margin), 3))
for era, g in d.groupby("era"):
    print(f"   {era}: open MAE {mae(-g.home_line_open, g.margin):.3f} | close MAE {mae(-g.home_line_close, g.margin):.3f} | n={len(g)}")
# Does the close beat the open? Bet at the OPEN the side the close favors.
for lo in [0.5, 1.0, 1.5, 2.0]:
    g = d[d.move.abs() >= lo]
    w, l, p = ats(g.home_line_close, g.home_line_open, g.margin)
    ci = stats.binomtest(w, w + l, 0.5).proportion_ci()
    print(f"   close-side bet AT THE OPEN, |move|>={lo}: {w}-{l}-{p} = {w/(w+l):.3f} (95% CI {ci.low:.3f}-{ci.high:.3f}), n={len(g)}")

print("\n(b) OLS: res_close ~ move  (coef<0 => steam continues past the close; coef>0 => market over-moves)")
for era in ["ALL", "fit<=2021", "test2022-25"]:
    g = d if era == "ALL" else d[d.era == era]
    X = sm.add_constant(g.move)
    r = sm.OLS(g.res_close, X).fit(cov_type="HC1")
    print(f"   {era:12s} n={len(g)} coef(move)={r.params['move']:+.3f} SE={r.bse['move']:.3f} p={r.pvalues['move']:.3f} | const={r.params['const']:+.3f}")
    # non-linear check: by move bucket
print("\n   residual vs close by move bucket (mean res_close, >0 = home beat close):")
d["mb"] = pd.cut(d.move, [-20, -2.75, -1.75, -0.75, -0.25, 0.25, 0.75, 1.75, 2.75, 20],
                 labels=["<=-3", "-2.5..-2", "-1.5..-1", "-0.5", "0", "+0.5", "+1..+1.5", "+2..+2.5", ">=+3"])
print(d.groupby(["mb"], observed=True).res_close.agg(n="size", mean="mean", se=lambda s: s.std() / np.sqrt(len(s))).round(3).to_string())

print("\n(c) ATS of betting WITH steam at the close (move<0 -> home; move>0 -> away):")
rows = []
for era in ["ALL", "fit<=2021", "test2022-25"]:
    for lo in [0.5, 1.0, 1.5, 2.0, 2.5]:
        g = d if era == "ALL" else d[d.era == era]
        g = g[g.move.abs() >= lo]
        # 'our number' = close moved one more step in the steam direction
        pred = g.home_line_close + np.sign(g.move) * 0.5
        w, l, p = ats(pred, g.home_line_close, g.margin)
        pb = stats.binomtest(w, w + l, 0.5).pvalue if (w + l) else np.nan
        rows.append(dict(era=era, min_abs_move=lo, n=len(g), W=w, L=l, P=p, rate=w / (w + l) if (w + l) else np.nan, p_binom=pb))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("   (FADE steam rate = 1 - rate)")

print("\n(d) Model pick (nfelo close vs market close) x steam agreement:")
g = d[d.nfelo_home_line_close.notna()].copy()
g["model_side"] = np.sign(g.home_line_close - g.nfelo_home_line_close)   # +1: nfelo likes home more than market (nfelo line more negative)
g["steam_side"] = -np.sign(g.move)                                         # +1: steam toward home
g = g[(g.model_side != 0)]
g["agree"] = np.where(g.steam_side == 0, "no move", np.where(g.model_side == g.steam_side, "steam agrees", "steam disagrees"))
rows = []
for era in ["ALL", "fit<=2021", "test2022-25"]:
    for a in ["no move", "steam agrees", "steam disagrees"]:
        h = g if era == "ALL" else g[g.era == era]
        h = h[h.agree == a]
        w, l, p = ats(h.nfelo_home_line_close, h.home_line_close, h.margin)
        pb = stats.binomtest(w, w + l, 0.5).pvalue if (w + l) else np.nan
        rows.append(dict(era=era, steam=a, n=len(h), W=w, L=l, P=p, model_pick_rate=w / (w + l) if (w + l) else np.nan, p_binom=pb))
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
# two-proportion z test agree vs disagree (all)
a1 = t[(t.era == "ALL") & (t.steam == "steam agrees")].iloc[0]; a2 = t[(t.era == "ALL") & (t.steam == "steam disagrees")].iloc[0]
p1, p2 = a1.W / (a1.W + a1.L), a2.W / (a2.W + a2.L); pp = (a1.W + a2.W) / (a1.W + a1.L + a2.W + a2.L)
z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / (a1.W + a1.L) + 1 / (a2.W + a2.L)))
print(f"   agree vs disagree difference (ALL): {p1-p2:+.3f}, z={z:.2f}, p={2*(1-stats.norm.cdf(abs(z))):.3f}")

print("\n(e) TOTALS open->close (only seasons with total_line_open):")
t2 = m[m.total_line_open.notna() & m.total_line_close.notna()].copy()
t2["move"] = t2.total_line_close - t2.total_line_open
t2["res"] = t2.total_pts - t2.total_line_close
print("   n =", len(t2), "| seasons:", sorted(t2.season.unique()), "| share no move:", round((t2.move == 0).mean(), 3), "| |move|>=1:", round((t2.move.abs() >= 1).mean(), 3))
print("   MAE open", round(mae(t2.total_line_open, t2.total_pts), 3), "| close", round(mae(t2.total_line_close, t2.total_pts), 3))
X = sm.add_constant(t2.move); r = sm.OLS(t2.res, X).fit(cov_type="HC1")
print(f"   OLS res_close ~ move: coef={r.params['move']:+.3f} SE={r.bse['move']:.3f} p={r.pvalues['move']:.3f}")
for lo in [0.5, 1.0, 1.5]:
    h = t2[t2.move.abs() >= lo]
    with_steam = np.sign(h.move) * np.sign(h.res)   # +1: total went the way it moved
    w = int((with_steam > 0).sum()); l = int((with_steam < 0).sum())
    print(f"   with-steam over/under at close, |move|>={lo}: {w}-{l} = {w/(w+l) if w+l else float('nan'):.3f}, n={len(h)}")
