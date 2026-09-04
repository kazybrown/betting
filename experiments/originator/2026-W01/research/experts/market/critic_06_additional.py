"""critic_06_additional.py - additional findings from the critic's re-analysis.
(1) Totals prior: the spec's league_prior = 46.0 is the 2025 realized MEAN total; the market total tracks the
    MEDIAN. Quantify the mean-median gap and what it does to an origin total vs the close.
(2) Odds-source note and juice structure by closing number (how much key-number value is priced in the juice).
(3) The 'ALL totals +0.51' and '|spread|>=10 +1.27' mean biases are skew, not median shifts: medians.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_06_additional.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.maximum(np.abs(o), 100))
m = merged(); m = m[m.mkt_spread.notna() & m.mkt_total.notna()].copy()

print("=== (1) totals prior: mean vs median ===")
for lab, g in [("2025", m[m.season == 2025]), ("2024", m[m.season == 2024]), ("2022-25", m[m.season >= 2022]), ("2009-25", m)]:
    e = g.total_err_mkt
    print(f"{lab}: n={len(g)} realized mean {g.total_pts.mean():.2f} median {g.total_pts.median():.1f} | mean closing line {g.mkt_total.mean():.2f} median line {g.mkt_total.median():.1f} | "
          f"mean err {e.mean():+.2f} (t p={stats.ttest_1samp(e, 0).pvalue:.3f}) median err {e.median():+.1f} | over {((e > 0).sum() / (e != 0).sum()):.3f} | skew {stats.skew(e):.2f}")
# what an origin total = market + delta does OOS 2022-25: over-rate and ROI of betting over when origin > market by >= 1
g = m[m.season >= 2022]; e = g.total_err_mkt
print("If the origin total sat +1.0 above the close (the mean-median gap), the engine would lean OVER; realized 2022-25 over rate at the close:",
      round((e > 0).sum() / (e != 0).sum(), 3), "| over ROI at actual prices:", round(np.where(e > 0, dec(g.over_odds) - 1, np.where(e < 0, -1.0, 0.0)).mean(), 3))
print("Recommended anchor = prior-season mean CLOSING total (median-targeting): 2024 ->", round(m[m.season == 2024].mkt_total.mean(), 2), "| 2025 ->", round(m[m.season == 2025].mkt_total.mean(), 2), "| 2022-25 ->", round(g.mkt_total.mean(), 2))

print("\n=== (2) juice by closing number (vig-free implied favorite prob), pooled and by odds-source era ===")
j = m[m.home_spread_odds.notna()].copy(); j["fav_line"] = j.mkt_spread.abs()
j["fav_odds"] = np.where(j.mkt_spread < 0, j.home_spread_odds, j.away_spread_odds); j["dog_odds"] = np.where(j.mkt_spread < 0, j.away_spread_odds, j.home_spread_odds)
j["p_fav"] = (1 / dec(j.fav_odds)) / (1 / dec(j.fav_odds) + 1 / dec(j.dog_odds)); j["era"] = np.where(j.season <= 2022, "2009-22", "2023-25")
j["vig"] = 1 / dec(j.fav_odds) + 1 / dec(j.dog_odds) - 1
print("overround by era:", j.groupby("era").vig.mean().round(3).to_dict())
t = j[j.fav_line.isin([2.5, 3, 3.5, 6.5, 7, 7.5, 9.5, 10, 10.5])].pivot_table(index="fav_line", columns="era", values="p_fav", aggfunc=["mean", "size"]).round(3)
print(t.to_string())
print("interpretation: p_fav<0.5 at 3.5/7.5/10 (dog side juiced) and >0.5 at 2.5/6.5 (fav juiced) is the market pricing part of the key-number value in the juice;")
print("the implied 'effective line' shift is ~ (p_fav-0.5)/mass_k points, e.g. at 3.5:", round((0.5 - j[(j.fav_line == 3.5)].p_fav.mean()) / 0.088, 2), "pts of the half point toward 3 (mass_3=0.088).")

print("\n=== (3) skew, not median shift ===")
for lab, e in [("ALL totals", m.total_err_mkt), ("|spread|>=10 spread err", m[m.mkt_spread.abs() >= 10].spread_err_mkt), ("dome totals", m[m.roof.eq('dome')].total_err_mkt)]:
    print(f"{lab}: n={len(e)} mean {e.mean():+.2f} median {e.median():+.1f} | sign test P(err>0) {((e > 0).sum() / (e != 0).sum()):.3f} (p={stats.binomtest(int((e > 0).sum()), int((e != 0).sum()), 0.5).pvalue:.3f})")

print("\n=== (4) MAE of (closing total + delta): where does the MAE-optimal anchor sit relative to the close? ===")
for lab, g in [("2022-25", m[m.season >= 2022]), ("2025", m[m.season == 2025]), ("2009-25", m)]:
    print(f"{lab}: " + " | ".join(f"delta {d:+.1f}: MAE {np.abs(g.total_pts - (g.mkt_total + d)).mean():.3f}" for d in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]))
print("=> MAE is minimised at delta<=0 (median-targeting); a +1 anchor (realized-mean prior vs the market's median line) costs ~0.05-0.1 pts of MAE and flags overs that hit ~49%.")
