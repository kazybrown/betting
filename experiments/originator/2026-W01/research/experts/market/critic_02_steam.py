"""critic_02_steam.py - adversarial re-analysis of T2 (open->close steam).
Attacks: (a) reproduce headline numbers; (b) robust specs: median (quantile) regression, Huber, and a
nonlinear 'fade big moves' test (|move|>=3), per 4-season block; (c) juice-aware steam: the with-steam side's
closing price (is the 51% rate at a worse price?), and price-only steam (implied-probability change open->close,
which captures moves that never change the number); (d) totals 2024-25 fade: realized ROI at actual close
prices, close-vs-open information content, split by |move| and by week.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_02_steam.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae, ats

def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.maximum(np.abs(o), 100))
def wilson(k, n):
    r = stats.binomtest(int(k), int(n), 0.5); c = r.proportion_ci(method="wilson"); return r.pvalue, c.low, c.high

m = merged()
d = m[m.home_line_open.notna() & m.home_line_close.notna() & m.margin.notna()].copy()
d["move"] = d.home_line_close - d.home_line_open; d["res_close"] = d.margin + d.home_line_close
d["block"] = pd.cut(d.season, [2008, 2012, 2016, 2021, 2025], labels=["2009-12", "2013-16", "2017-21", "2022-25"])
print("n =", len(d), "| sign check corr(home_line_close, margin) =", round(np.corrcoef(d.home_line_close, d.margin)[0, 1], 3))

print("\n=== (a) reproduce: OLS res_close ~ move (HC1) and with-steam ATS at |move|>=1 ===")
r = sm.OLS(d.res_close, sm.add_constant(d.move)).fit(cov_type="HC1"); print(f"OLS coef {r.params['move']:+.3f} SE {r.bse['move']:.3f} p {r.pvalues['move']:.3f}")
g = d[d.move.abs() >= 1]; w, l, p = ats(g.home_line_close + np.sign(g.move) * 0.5, g.home_line_close, g.margin)
print(f"with-steam ATS |move|>=1: {w}-{l}-{p} = {w/(w+l):.4f} (p={wilson(w, w+l)[0]:.3f}), n={len(g)}")

print("\n=== (b) robust specs ===")
q = QuantReg(d.res_close, sm.add_constant(d.move)).fit(q=0.5); print(f"median regression coef(move) {q.params['move']:+.3f} SE {q.bse['move']:.3f} p {q.pvalues['move']:.3f}")
h = sm.RLM(d.res_close, sm.add_constant(d.move), M=sm.robust.norms.HuberT()).fit(); print(f"Huber coef(move) {h.params['move']:+.3f} SE {h.bse['move']:.3f} p {h.pvalues['move']:.3f}")
print("with-steam ATS by |move| bucket and 4-season block (rate = with-steam wins share):")
rows = []
for lo, hi in [(0.5, 0.75), (1.0, 1.75), (2.0, 2.75), (3.0, 99)]:
    for blk, gb in d.groupby("block", observed=True):
        g = gb[(gb.move.abs() >= lo) & (gb.move.abs() <= hi)]
        w, l, p = ats(g.home_line_close + np.sign(g.move) * 0.5, g.home_line_close, g.margin)
        rows.append(dict(abs_move=f"{lo}-{hi}", block=blk, n=len(g), W=w, L=l, rate=w / (w + l) if w + l else np.nan))
    g = d[(d.move.abs() >= lo) & (d.move.abs() <= hi)]
    w, l, p = ats(g.home_line_close + np.sign(g.move) * 0.5, g.home_line_close, g.margin); pv, lo_, hi_ = wilson(w, w + l)
    rows.append(dict(abs_move=f"{lo}-{hi}", block="ALL", n=len(g), W=w, L=l, rate=w / (w + l), p=pv, ci_lo=lo_, ci_hi=hi_))
print(pd.DataFrame(rows).round(3).to_string(index=False))
g = d[d.move.abs() >= 3]; w, l, p = ats(g.home_line_close + np.sign(g.move) * 0.5, g.home_line_close, g.margin); pv, lo_, hi_ = wilson(l, w + l)
print(f"FADE big moves (|move|>=3): fade wins {l}-{w} = {l/(w+l):.3f} (95% CI {lo_:.3f}-{hi_:.3f}, p={pv:.3f}), n={len(g)} | fit era: ", end="")
gf = g[g.season <= 2021]; w2, l2, _ = ats(gf.home_line_close + np.sign(gf.move) * 0.5, gf.home_line_close, gf.margin); print(f"{l2/(w2+l2):.3f} (n={len(gf)}) | test: ", end="")
gt = g[g.season >= 2022]; w3, l3, _ = ats(gt.home_line_close + np.sign(gt.move) * 0.5, gt.home_line_close, gt.margin); print(f"{l3/(w3+l3):.3f} (n={len(gt)})")

print("\n=== (c) juice-aware steam ===")
j = d[d.home_spread_odds.notna() & d.away_spread_odds.notna() & (d.move != 0)].copy()
j["steam_home"] = j.move < 0
j["steam_odds"] = np.where(j.steam_home, j.home_spread_odds, j.away_spread_odds); j["other_odds"] = np.where(j.steam_home, j.away_spread_odds, j.home_spread_odds)
j["p_steam_imp"] = (1 / dec(j.steam_odds)) / (1 / dec(j.steam_odds) + 1 / dec(j.other_odds))
j["steam_cover"] = np.sign(j.res_close) * np.where(j.steam_home, 1, -1)
for lo in [0.5, 1.0, 2.0]:
    g = j[j.move.abs() >= lo]; s = g[g.steam_cover != 0]; k = int((s.steam_cover > 0).sum())
    ret = np.where(s.steam_cover > 0, dec(s.steam_odds) - 1, -1.0)
    print(f"|move|>={lo}: n={len(s)} with-steam cover {k/len(s):.3f} | implied from closing juice {s.p_steam_imp.mean():.3f} | mean steam-side odds {s.steam_odds.mean():.0f} vs other {s.other_odds.mean():.0f} | ROI with steam at actual price {ret.mean():+.3f}")
# price-only steam: implied home cover prob change open->close from nfelo prices (number may not move)
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
n["gid"] = n.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").str.replace("_STL_", "_LA_").str.replace("_SD_", "_LAC_")
n = n[["gid", "home_line_open_price", "away_line_open_price", "home_line_close_price", "away_line_close_price"]]
jj = d.merge(n, on="gid", how="left").dropna(subset=["home_line_open_price", "home_line_close_price", "away_line_open_price", "away_line_close_price"])
print("games with open & close PRICES from nfelo:", len(jj), "| seasons", jj.season.min(), "-", jj.season.max())
def ph(hp, ap):
    return (1 / dec(hp)) / (1 / dec(hp) + 1 / dec(ap))
jj["dp"] = ph(jj.home_line_close_price, jj.away_line_close_price) - ph(jj.home_line_open_price, jj.away_line_open_price)   # >0: home side got more expensive at the SAME number
nm = jj[jj.move == 0].copy()
print(f"no-number-move games: {len(nm)} | of which price moved (|dp|>=0.01): {int((nm.dp.abs() >= 0.01).sum())}")
for lo in [0.01, 0.02]:
    g = nm[nm.dp.abs() >= lo]; s = g[g.res_close != 0]
    with_steam = np.sign(s.res_close) * np.sign(s.dp); k = int((with_steam > 0).sum())
    pv, lo_, hi_ = wilson(k, len(s)); print(f"  price-steam (|dp|>={lo}) with-steam cover at close: {k}/{len(s)} = {k/len(s):.3f} (CI {lo_:.3f}-{hi_:.3f}, p={pv:.3f})")
# combined signal: implied-prob change from number+price (use nfelo effective line: line + juice), regress residual on total implied change
jj["imp_open"] = ph(jj.home_line_open_price, jj.away_line_open_price); jj["imp_close"] = ph(jj.home_line_close_price, jj.away_line_close_price)
X = sm.add_constant(pd.DataFrame({"move": jj.move, "dp": jj.dp})); r = sm.OLS(jj.res_close, X).fit(cov_type="HC1")
print(f"OLS res_close ~ move + dp: coef(move) {r.params['move']:+.3f} (p={r.pvalues['move']:.3f}), coef(dp) {r.params['dp']:+.2f} (p={r.pvalues['dp']:.3f}), n={len(jj)}")

print("\n=== (d) totals 2024-25: fade the move ===")
t2 = m[m.total_line_open.notna() & m.total_line_close.notna()].copy()
t2["move"] = t2.total_line_close - t2.total_line_open; t2["res"] = t2.total_pts - t2.total_line_close
t2 = t2[t2.season >= 2024]
print("n =", len(t2), "| seasons", sorted(t2.season.unique()), "| corr(total_line_close, total_pts) =", round(np.corrcoef(t2.total_line_close, t2.total_pts)[0, 1], 3))
print("MAE open", round(mae(t2.total_line_open, t2.total_pts), 3), "| close", round(mae(t2.total_line_close, t2.total_pts), 3))
g = t2[t2.move.abs() >= 0.5]; cs = np.sign(g.move) * np.sign(g.total_pts - g.total_line_open)   # close-side bet at the open
w = int((cs > 0).sum()); l = int((cs < 0).sum()); pv, lo_, hi_ = wilson(w, w + l)
print(f"close-side bet AT THE OPEN (info in the move vs opener): {w}-{l} = {w/(w+l):.3f} (CI {lo_:.3f}-{hi_:.3f}, p={pv:.3f})  [spreads: 56%]")
for lo in [0.5, 1.0, 1.5, 2.0]:
    g = t2[t2.move.abs() >= lo]; s = g[g.res != 0]; fade = np.sign(s.res) != np.sign(s.move); k = int(fade.sum())
    fade_odds = np.where(s.move > 0, s.under_odds, s.over_odds)     # move up -> fade = under
    ret = np.where(fade, dec(fade_odds) - 1, -1.0); pv, lo_, hi_ = wilson(k, len(s))
    print(f"|move|>={lo}: n={len(s)} fade wins {k/len(s):.3f} (CI {lo_:.3f}-{hi_:.3f}, p={pv:.3f}) | mean fade-side odds {np.nanmean(fade_odds):.0f} | ROI at actual price {np.nanmean(ret):+.3f} | by season: " +
          str(s.assign(f=fade).groupby("season").f.agg(lambda x: f"{x.mean():.3f}(n={len(x)})").to_dict()))
g = t2[t2.move.abs() >= 0.5]; s = g[g.res != 0]; s = s.assign(fade=np.sign(s.res) != np.sign(s.move), half=np.where(s.week <= 9, "wk1-9", "wk10+"))
print("fade rate by half-season:", s.groupby("half").fade.agg(["mean", "size"]).round(3).to_dict())
print("fade rate by direction: move UP (fade=under):", round(s[s.move > 0].fade.mean(), 3), f"(n={int((s.move>0).sum())})", "| move DOWN (fade=over):", round(s[s.move < 0].fade.mean(), 3), f"(n={int((s.move<0).sum())})")
r = sm.OLS(t2.res, sm.add_constant(t2.move)).fit(cov_type="HC1"); print(f"OLS res_close ~ move: {r.params['move']:+.3f} SE {r.bse['move']:.3f} p {r.pvalues['move']:.3f}")
q = QuantReg(t2.res, sm.add_constant(t2.move)).fit(q=0.5); print(f"median regression: {q.params['move']:+.3f} SE {q.bse['move']:.3f} p {q.pvalues['move']:.3f}")
