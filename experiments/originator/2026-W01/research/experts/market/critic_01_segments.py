"""critic_01_segments.py - adversarial re-analysis of T1 (segment efficiency of the closing line).
Attacks: (a) multiple-comparison correction (Holm / BH) over all 48 segments; (b) placebo segments of the
same sizes drawn at random -> how many p<0.05 hits does pure noise produce; (c) per-season sign persistence
of every cell the expert flagged; (d) juice-aware check: implied cover probability from the closing spread
odds vs realized cover rate per segment (a 'bias' that is already priced in the juice is not exploitable);
(e) wind>=15 unders in detail; (f) totals mean vs median vs the league_prior=46.0 in the spec.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_01_segments.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, mae
rng = np.random.default_rng(11)

m = merged()
m = m[m.mkt_spread.notna() & m.mkt_total.notna()].copy()
m["abs_spread"] = m.mkt_spread.abs()
m["home_fav"] = m.mkt_spread < 0; m["away_fav"] = m.mkt_spread > 0
m["hour"] = m.gametime.str.slice(0, 2).astype(float)
m["primetime"] = (m.hour >= 19) | m.weekday.isin(["Monday", "Thursday"])
m["snf"] = m.weekday.eq("Sunday") & (m.hour >= 19); m["mnf"] = m.weekday.eq("Monday"); m["tnf"] = m.weekday.eq("Thursday")
m["week1"] = (m.week == 1) & (m.game_type == "REG"); m["playoff"] = m.game_type != "REG"
m["late_season"] = (m.game_type == "REG") & (m.week >= 15); m["is_div"] = m.div_game == 1
m["era"] = np.where(m.season <= 2021, "fit", "test")
m["tot_bin"] = pd.cut(m.mkt_total, [0, 40, 43.5, 46.5, 49.5, 53, 99], labels=["<=40", "40.5-43.5", "44-46.5", "47-49.5", "50-53", "53.5+"])
print("sanity: corr(mkt_spread, margin) =", round(np.corrcoef(m.mkt_spread, m.margin)[0, 1], 3), "| n =", len(m))

spread_segs = {
    "ALL": m.index == m.index, "home fav": m.home_fav, "away fav": m.away_fav,
    "|spread| 0-2.5": m.abs_spread <= 2.5, "|spread| 3-6.5": (m.abs_spread >= 3) & (m.abs_spread <= 6.5),
    "|spread| 7-9.5": (m.abs_spread >= 7) & (m.abs_spread <= 9.5), "|spread| 10+": m.abs_spread >= 10, "|spread| 14+": m.abs_spread >= 14,
    "home fav 0-2.5": m.home_fav & (m.abs_spread <= 2.5), "home fav 3-6.5": m.home_fav & (m.abs_spread >= 3) & (m.abs_spread <= 6.5),
    "home fav 7-9.5": m.home_fav & (m.abs_spread >= 7) & (m.abs_spread <= 9.5), "home fav 10+": m.home_fav & (m.abs_spread >= 10),
    "away fav 0-2.5": m.away_fav & (m.abs_spread <= 2.5), "away fav 3-6.5": m.away_fav & (m.abs_spread >= 3) & (m.abs_spread <= 6.5),
    "away fav 7+": m.away_fav & (m.abs_spread >= 7),
    "primetime": m.primetime, "SNF": m.snf, "MNF": m.mnf, "TNF": m.tnf, "Sun day": ~m.primetime & m.weekday.eq("Sunday"),
    "divisional": m.is_div, "non-div": ~m.is_div, "week 1": m.week1, "weeks 2-14": (m.game_type == "REG") & (m.week >= 2) & (m.week <= 14),
    "weeks 15+": m.late_season, "playoffs": m.playoff, "neutral": m.neutral, "dome/closed": m.is_dome, "outdoors": ~m.is_dome,
}
tot_segs = {"ALL": m.index == m.index}
tot_segs.update({f"total {b}": m.tot_bin == b for b in m.tot_bin.cat.categories})
tot_segs.update({"roof outdoors": m.roof.eq("outdoors"), "roof dome": m.roof.eq("dome"), "roof closed": m.roof.eq("closed"), "roof open": m.roof.eq("open"),
                 "primetime": m.primetime, "divisional": m.is_div, "week 1": m.week1, "playoffs": m.playoff, "weeks 15+": m.late_season,
                 "wind>=15": m.wind >= 15, "temp<32": m.temp < 32})

# ---------- (a) multiple comparisons ----------
rows = []
for name, mk in spread_segs.items():
    e = m.loc[mk, "spread_err_mkt"]; rows.append(dict(kind="spread", seg=name, n=len(e), bias=e.mean(), p=stats.ttest_1samp(e, 0).pvalue))
for name, mk in tot_segs.items():
    e = m.loc[mk, "total_err_mkt"]; rows.append(dict(kind="total", seg=name, n=len(e), bias=e.mean(), p=stats.ttest_1samp(e, 0).pvalue))
t = pd.DataFrame(rows).dropna(); t = t[t.n >= 20]
from statsmodels.stats.multitest import multipletests
t["p_holm"] = multipletests(t.p, method="holm")[1]; t["p_bh"] = multipletests(t.p, method="fdr_bh")[1]
print("\n=== (a) All-era segment biases with Holm and BH corrected p (K =", len(t), "tests) ===")
print(t.sort_values("p").head(12).round(4).to_string(index=False))
print("raw p<0.05:", int((t.p < 0.05).sum()), "| Holm p<0.05:", int((t.p_holm < 0.05).sum()), "| BH q<0.10:", int((t.p_bh < 0.10).sum()))

# ---------- (b) placebo segments ----------
sizes = t.n.values; K = len(sizes)
real_hits = int((t.p < 0.05).sum()); real_min = t.p.min()
hits, mins = [], []
for _ in range(2000):
    h, mn = 0, 1.0
    for n, kind in zip(sizes, t.kind.values):
        idx = rng.choice(len(m), size=int(n), replace=False)
        e = (m.spread_err_mkt if kind == "spread" else m.total_err_mkt).values[idx]
        p = stats.ttest_1samp(e, 0).pvalue; h += p < 0.05; mn = min(mn, p)
    hits.append(h); mins.append(mn)
hits = np.array(hits); mins = np.array(mins)
print(f"\n=== (b) Placebo: {K} random segments with the real sizes, 2000 reps ===")
print(f"real: {real_hits} hits at p<0.05, min p = {real_min:.4f} | placebo hits mean {hits.mean():.2f}, 95th pct {np.percentile(hits, 95):.0f}, P(placebo hits >= real) = {(hits >= real_hits).mean():.3f}")
print(f"placebo min-p: median {np.median(mins):.4f}, P(placebo min p <= real min p) = {(mins <= real_min).mean():.3f}")
print("(note: ALL-era 'ALL' total bias +0.51 is one of the real hits; random segments include it as a near-duplicate of the population so the placebo is conservative)")

# ---------- (c) per-season persistence of flagged cells ----------
print("\n=== (c) Per-season sign persistence of the flagged cells (share of seasons with the claimed sign; binomial p vs 0.5) ===")
flag = [("spread", "home fav 0-2.5", spread_segs["home fav 0-2.5"], -1), ("spread", "|spread| 10+", spread_segs["|spread| 10+"], +1),
        ("spread", "SNF", spread_segs["SNF"], +1), ("total", "roof dome", tot_segs["roof dome"], +1), ("total", "roof closed", tot_segs["roof closed"], +1),
        ("total", "wind>=15", tot_segs["wind>=15"], -1), ("total", "ALL", tot_segs["ALL"], +1)]
for kind, name, mk, sgn in flag:
    col = "spread_err_mkt" if kind == "spread" else "total_err_mkt"
    bys = m[mk].groupby("season")[col].agg(["mean", "size"])
    k = int((np.sign(bys["mean"]) == sgn).sum()); n = len(bys)
    # cover-rate version: share of seasons where the natural side > 50%
    print(f"{kind:6s} {name:16s} seasons with sign {sgn:+d}: {k}/{n} (p={stats.binomtest(k, n, 0.5).pvalue:.3f}) | per-season means: {bys['mean'].round(1).to_dict()}")

# ---------- (d) juice-aware: implied vs realized home cover ----------
def imp(o):
    o = np.asarray(o, float); return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
m["p_home_imp"] = imp(m.home_spread_odds) / (imp(m.home_spread_odds) + imp(m.away_spread_odds))   # vig-free implied home cover prob
m["p_over_imp"] = imp(m.over_odds) / (imp(m.over_odds) + imp(m.under_odds))
print("\n=== (d) Juice-aware: vig-free implied home-cover prob vs realized (excl pushes); a bias already in the price is not exploitable ===")
print("mean home_spread_odds", round(m.home_spread_odds.mean(), 1), "| away", round(m.away_spread_odds.mean(), 1), "| implied home cover mean", round(m.p_home_imp.mean(), 4))
rows = []
for name, mk in spread_segs.items():
    d = m[mk & (m.spread_err_mkt != 0)]
    if len(d) < 50: continue
    real = (d.spread_err_mkt > 0).mean(); impl = d.p_home_imp.mean()
    se = np.sqrt(real * (1 - real) / len(d))
    # ROI of betting home at the actual price, and of betting the favorite
    ret_home = np.where(d.spread_err_mkt > 0, np.where(d.home_spread_odds > 0, d.home_spread_odds / 100, 100 / -d.home_spread_odds), -1.0)
    rows.append(dict(seg=name, n=len(d), home_cover=real, implied=impl, excess=real - impl, z=(real - impl) / se, roi_home=ret_home.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
rows = []
for name, mk in tot_segs.items():
    d = m[mk & (m.total_err_mkt != 0)]
    if len(d) < 50: continue
    real = (d.total_err_mkt > 0).mean(); impl = d.p_over_imp.mean(); se = np.sqrt(real * (1 - real) / len(d))
    ret_under = np.where(d.total_err_mkt < 0, np.where(d.under_odds > 0, d.under_odds / 100, 100 / -d.under_odds), -1.0)
    rows.append(dict(seg=name, n=len(d), over_rate=real, implied_over=impl, excess=real - impl, z=(real - impl) / se, roi_under=ret_under.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- (e) wind ----------
print("\n=== (e) wind>=15 unders in detail ===")
w = m[m.wind.notna()].copy()
print("wind coverage (outdoor games with wind):", len(w), "| wind>=15:", int((w.wind >= 15).sum()), "| wind>=20:", int((w.wind >= 20).sum()))
print("market total line by wind bin (does the market already price wind?):")
print(w.groupby(pd.cut(w.wind, [-1, 4, 9, 14, 19, 60])).agg(n=("total_err_mkt", "size"), line=("mkt_total", "mean"), pts=("total_pts", "mean"), err=("total_err_mkt", "mean"), under=("total_err_mkt", lambda s: (s < 0).sum() / (s != 0).sum())).round(2).to_string())
import statsmodels.api as sm
for era in ["fit", "test", "ALL"]:
    d = w if era == "ALL" else w[w.era == era]
    X = sm.add_constant(np.clip(d.wind - 10, 0, None)); r = sm.OLS(d.total_err_mkt, X).fit(cov_type="HC1")
    print(f"  residual ~ max(wind-10,0)  [{era}] n={len(d)} slope={r.params.iloc[1]:+.3f}/mph SE={r.bse.iloc[1]:.3f} p={r.pvalues.iloc[1]:.3f}")
for lo in [15, 20]:
    d = w[w.wind >= lo]; s = d.total_err_mkt[d.total_err_mkt != 0]; k = int((s < 0).sum())
    bys = d.groupby("season").total_err_mkt.apply(lambda s: (s < 0).sum() / max((s != 0).sum(), 1))
    print(f"  wind>={lo}: n={len(d)} bias={d.total_err_mkt.mean():+.2f} under={k/len(s):.3f} (p={stats.binomtest(k, len(s), 0.5).pvalue:.3f}) | seasons under>50%: {int((bys > 0.5).sum())}/{len(bys)} | implied under from juice {1 - d.p_over_imp.mean():.3f}")

# ---------- (f) totals mean vs median vs the spec's league_prior ----------
print("\n=== (f) Totals: mean vs median of realized total, vs mean closing total (spec league_prior = 46.0 = '2025 realized mean') ===")
g = m.groupby("season").agg(n=("total_pts", "size"), mean_pts=("total_pts", "mean"), median_pts=("total_pts", "median"), mean_line=("mkt_total", "mean"),
                            mean_err=("total_err_mkt", "mean"), median_err=("total_err_mkt", "median"), over_rate=("total_err_mkt", lambda s: (s > 0).sum() / (s != 0).sum()))
print(g.round(2).to_string())
print("2022-25 pooled: mean_pts - mean_line =", round((m[m.era == "test"].total_pts - m[m.era == "test"].mkt_total).mean(), 2),
      "| median err =", m[m.era == "test"].total_err_mkt.median(), "| skew of total err =", round(stats.skew(m.total_err_mkt), 3))
print("=> a prior anchored to the realized MEAN sits above the market's median-targeting line by the mean-median gap; see verdict notes.")
