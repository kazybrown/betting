"""01_segment_efficiency.py - Theory 1: where does the closing line miss systematically?
Spread error = margin + mkt_spread (0 = market right; >0 = home did better than the line).
Total error = total_pts - mkt_total (>0 = over).
For each segment: n, mean error (bias) with 95% CI and t-test p, MAE, ATS rate of the
segment's 'natural' side (home / favorite / over), and the SAME stats split by
fit era (2009-2021) vs test era (2022-2025) so we can see whether a bias persists OOS.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/01_segment_efficiency.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae

m = merged()
m = m[m.mkt_spread.notna() & m.mkt_total.notna()].copy()
m["abs_spread"] = m.mkt_spread.abs()
m["home_fav"] = m.mkt_spread < 0
m["away_fav"] = m.mkt_spread > 0
m["pick"] = m.mkt_spread == 0
m["hour"] = m.gametime.str.slice(0, 2).astype(float)
m["primetime"] = (m.hour >= 19) | m.weekday.isin(["Monday", "Thursday"])
m["mnf"] = m.weekday.eq("Monday")
m["tnf"] = m.weekday.eq("Thursday")
m["snf"] = m.weekday.eq("Sunday") & (m.hour >= 19)
m["week1"] = (m.week == 1) & (m.game_type == "REG")
m["playoff"] = m.game_type != "REG"
m["late_season"] = (m.game_type == "REG") & (m.week >= 15)
m["is_div"] = m.div_game == 1
m["era"] = np.where(m.season <= 2021, "fit<=2021", "test2022-25")
# favorite cover: home covers if err>0; favorite covers if (home_fav & err>0) | (away_fav & err<0)
m["home_cover"] = np.sign(m.spread_err_mkt)          # +1 home covers, -1 away covers, 0 push
m["fav_cover"] = np.where(m.home_fav, m.home_cover, np.where(m.away_fav, -m.home_cover, np.nan))
m["over"] = np.sign(m.total_err_mkt)

def row(d, err, side):
    d = d.dropna(subset=[err])
    n = len(d); e = d[err]
    if n < 2:
        return dict(n=n)
    t, p = stats.ttest_1samp(e, 0.0)
    se = e.std(ddof=1) / np.sqrt(n)
    s = d[side].dropna(); s = s[s != 0]
    rate = (s > 0).mean() if len(s) else np.nan
    # binomial exact-ish CI on the ATS rate
    k = int((s > 0).sum()); nn = len(s)
    pb = stats.binomtest(k, nn, 0.5).pvalue if nn else np.nan
    return dict(n=n, bias=e.mean(), ci_lo=e.mean() - 1.96 * se, ci_hi=e.mean() + 1.96 * se, p_t=p,
                mae=e.abs().mean(), ats_rate=rate, ats_n=nn, p_binom=pb)

def table(segs, err, side, title):
    rows = []
    for name, mask in segs:
        for era in ["ALL", "fit<=2021", "test2022-25"]:
            d = m[mask] if era == "ALL" else m[mask & (m.era == era)]
            r = row(d, err, side); r.update(segment=name, era=era); rows.append(r)
    t = pd.DataFrame(rows).set_index(["segment", "era"])
    print(f"\n=== {title} ===  (side rate = {side} > 0 share, excl. pushes)")
    print(t.round(3).to_string())
    return t

spread_segs = [
    ("ALL", m.index == m.index),
    ("home fav", m.home_fav), ("away fav", m.away_fav), ("pick", m.pick),
    ("|spread| 0-2.5", m.abs_spread <= 2.5), ("|spread| 3-6.5", (m.abs_spread >= 3) & (m.abs_spread <= 6.5)),
    ("|spread| 7-9.5", (m.abs_spread >= 7) & (m.abs_spread <= 9.5)), ("|spread| 10+", m.abs_spread >= 10),
    ("|spread| 14+", m.abs_spread >= 14),
    ("home fav 0-2.5", m.home_fav & (m.abs_spread <= 2.5)), ("home fav 3-6.5", m.home_fav & (m.abs_spread >= 3) & (m.abs_spread <= 6.5)),
    ("home fav 7-9.5", m.home_fav & (m.abs_spread >= 7) & (m.abs_spread <= 9.5)), ("home fav 10+", m.home_fav & (m.abs_spread >= 10)),
    ("away fav 0-2.5", m.away_fav & (m.abs_spread <= 2.5)), ("away fav 3-6.5", m.away_fav & (m.abs_spread >= 3) & (m.abs_spread <= 6.5)),
    ("away fav 7+", m.away_fav & (m.abs_spread >= 7)),
    ("primetime (any)", m.primetime), ("SNF", m.snf), ("MNF", m.mnf), ("TNF", m.tnf), ("Sun day games", ~m.primetime & m.weekday.eq("Sunday")),
    ("divisional", m.is_div), ("non-div", ~m.is_div),
    ("week 1", m.week1), ("weeks 2-14", (m.game_type == "REG") & (m.week >= 2) & (m.week <= 14)), ("weeks 15+", m.late_season),
    ("playoffs", m.playoff), ("neutral site", m.neutral),
    ("dome/closed", m.is_dome), ("outdoors", ~m.is_dome),
]
ts = table(spread_segs, "spread_err_mkt", "home_cover", "SPREAD: closing-line error by segment (bias>0 = home beat the line)")
# favorite view for the size bins
fav_segs = [(n, mk) for n, mk in spread_segs if "spread|" in n or n in ("home fav", "away fav", "primetime (any)", "playoffs", "week 1", "divisional")]
tf = table(fav_segs, "spread_err_mkt", "fav_cover", "SPREAD: favorite cover rates by segment (ats_rate = favorite covers)")

m["tot_bin"] = pd.cut(m.mkt_total, [0, 40, 43.5, 46.5, 49.5, 53, 99], labels=["<=40", "40.5-43.5", "44-46.5", "47-49.5", "50-53", "53.5+"])
tot_segs = [("ALL", m.index == m.index)] + [(f"total {b}", m.tot_bin == b) for b in m.tot_bin.cat.categories] + [
    ("roof outdoors", m.roof.eq("outdoors")), ("roof dome", m.roof.eq("dome")), ("roof closed", m.roof.eq("closed")), ("roof open", m.roof.eq("open")),
    ("primetime", m.primetime), ("divisional", m.is_div), ("week 1", m.week1), ("playoffs", m.playoff), ("weeks 15+", m.late_season),
    ("wind>=15 (outdoor)", m.wind >= 15), ("temp<32 (outdoor)", m.temp < 32),
]
tt = table(tot_segs, "total_err_mkt", "over", "TOTAL: closing-line error by segment (bias>0 = over; ats_rate = over rate)")

# Multiple-comparison guard: how many of the spread segments have |t|>2 in BOTH eras with the same sign?
print("\n=== Persistence check: segments where fit-era bias and test-era bias share sign and fit-era p<0.05 ===")
for name, _ in spread_segs:
    a, b = ts.loc[(name, "fit<=2021")], ts.loc[(name, "test2022-25")]
    if a.get("p_t", 1) < 0.05:
        print(f"spread  {name:18s} fit bias {a.bias:+.2f} (p={a.p_t:.3f}, n={int(a.n)}) | test bias {b.bias:+.2f} (p={b.p_t:.3f}, n={int(b.n)}) | same sign: {np.sign(a.bias)==np.sign(b.bias)}")
for name, _ in tot_segs:
    a, b = tt.loc[(name, "fit<=2021")], tt.loc[(name, "test2022-25")]
    if a.get("p_t", 1) < 0.05:
        print(f"total   {name:18s} fit bias {a.bias:+.2f} (p={a.p_t:.3f}, n={int(a.n)}) | test bias {b.bias:+.2f} (p={b.p_t:.3f}, n={int(b.n)}) | same sign: {np.sign(a.bias)==np.sign(b.bias)}")

# Regression check: is |error| (MAE) related to spread size? (heteroskedasticity, for confidence tags)
print("\n=== Spread |error| by |spread| (for confidence tags) ===")
print(m.groupby(pd.cut(m.abs_spread, [-1, 2.5, 6.5, 9.5, 13.5, 30])).spread_err_mkt.agg(n="size", mean="mean", mae=lambda s: s.abs().mean(), sd="std").round(2).to_string())
print("\n=== Total |error| by total size ===")
print(m.groupby("tot_bin", observed=True).total_err_mkt.agg(n="size", mean="mean", mae=lambda s: s.abs().mean(), sd="std").round(2).to_string())
