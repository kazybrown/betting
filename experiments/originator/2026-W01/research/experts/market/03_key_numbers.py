"""03_key_numbers.py - Theories 3 & 4: margin distribution at key numbers, push probabilities,
half-point value table around 3 and 7, and an empirical backtest of 'a half point crossing a key
number is worth more' using nfelo-close vs market-close disagreements.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/03_key_numbers.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged, ats

m = merged()
m = m[m.mkt_spread.notna()].copy()
m["era"] = np.where(m.season <= 2021, "fit<=2021", "test2022-25")
# favorite frame: fav_line = |spread| (points the favorite lays), fav_margin = favorite's actual margin
m["fav_line"] = m.mkt_spread.abs()
m["fav_margin"] = np.where(m.mkt_spread < 0, m.margin, np.where(m.mkt_spread > 0, -m.margin, m.margin))
m["abs_margin"] = m.margin.abs()

def ci(k, n):
    r = stats.binomtest(int(k), int(n), 0.5).proportion_ci(method="wilson"); return r.low, r.high

print("n games:", len(m))
print("\n=== (1) Unconditional distribution of |margin| (all games 2009-2025) ===")
rows = []
for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 21]:
    n = len(m); c = int((m.abs_margin == k).sum()); lo, hi = ci(c, n)
    c1 = int(((m.abs_margin == k) & (m.era == "fit<=2021")).sum()); n1 = int((m.era == "fit<=2021").sum())
    c2 = int(((m.abs_margin == k) & (m.era == "test2022-25")).sum()); n2 = int((m.era == "test2022-25").sum())
    rows.append(dict(abs_margin=k, share=c / n, ci_lo=lo, ci_hi=hi, n=c, share_fit=c1 / n1, share_test=c2 / n2))
t1 = pd.DataFrame(rows); print(t1.round(4).to_string(index=False))
print("rank of |margin| values by frequency:", m.abs_margin.value_counts().head(8).to_dict())
print("share of games decided by exactly 3 or 7:", round(m.abs_margin.isin([3, 7]).mean(), 4))

print("\n=== (2) Push probability at the market line: P(fav_margin == line | fav_line == k) ===")
rows = []
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]:
    g = m[m.fav_line == k]; c = int((g.fav_margin == k).sum())
    if len(g):
        lo, hi = ci(c, len(g))
        rows.append(dict(line=k, n=len(g), pushes=c, push_prob=c / len(g), ci_lo=lo, ci_hi=hi,
                         fav_cover=(g.fav_margin > k).mean(), dog_cover=(g.fav_margin < k).mean()))
t2 = pd.DataFrame(rows); print(t2.round(3).to_string(index=False))

print("\n=== (3) 'Landing' probability, pooled window: P(fav_margin == k | fav_line within +/-2 of k) ===")
rows = []
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
    g = m[(m.fav_line >= k - 2) & (m.fav_line <= k + 2)]; c = int((g.fav_margin == k).sum())
    lo, hi = ci(c, len(g))
    g1 = g[g.era == "fit<=2021"]; g2 = g[g.era == "test2022-25"]
    rows.append(dict(k=k, n=len(g), land_prob=c / len(g), ci_lo=lo, ci_hi=hi,
                     land_fit=(g1.fav_margin == k).mean(), land_test=(g2.fav_margin == k).mean(),
                     # also probability of landing on -k (favorite loses by k), relevant when line is small
                     land_minus_k=(g.fav_margin == -k).mean()))
t3 = pd.DataFrame(rows); print(t3.round(3).to_string(index=False))

print("\n=== (4) Empirical cover table by exact market number (favorite frame), pooled both eras ===")
rows = []
for k in np.arange(0.5, 14.1, 0.5):
    g = m[m.fav_line == k]
    if len(g) < 25: continue
    w = (g.fav_margin > k).mean(); p = (g.fav_margin == k).mean(); l = (g.fav_margin < k).mean()
    rows.append(dict(line=k, n=len(g), fav_win=w, push=p, dog_win=l, fav_ev_110=(w - 1.1 * l) / 1.1, dog_ev_110=(l - 1.1 * w) / 1.1))
t4 = pd.DataFrame(rows); print(t4.round(3).to_string(index=False))

print("\n=== (5) HALF-POINT VALUE TABLE (favorite frame). Value of moving the favorite's line from a to b ===")
print("Method: the cover-probability change from crossing integer k equals the landing mass at k.")
print("Using pooled landing probabilities (table 3) as the mass; value in cover-prob points and in")
print("EV per unit risked at -110 for the side that gains (win +1 / push 0 / loss -1.1, divided by 1.1).")
land = dict(zip(t3.k, t3.land_prob))
rows = []
for k in range(1, 15):
    pk = land[k]
    # from k-0.5 to k: favorite's wins at margin==k become pushes: fav loses pk in win prob (EV -pk/1.1 per unit)
    # from k to k+0.5: pushes become losses: fav loses pk in EV as 1.1*pk/1.1 = pk
    rows.append(dict(cross=k, mass=pk, half_pt_to_key=f"{k-0.5}->{k}", cover_prob_change=pk, ev_change_per_unit=pk / 1.1,
                     half_pt_from_key=f"{k}->{k+0.5}", ev_change_per_unit_2=pk * 1.1 / 1.1))
t5 = pd.DataFrame(rows); print(t5.round(3).to_string(index=False))
generic = t5[~t5.cross.isin([3, 7])].mass.mean()
print(f"\nmean mass at non-key integers 1-14 (excl 3,7): {generic:.3f} | at 3: {land[3]:.3f} ({land[3]/generic:.1f}x) | at 7: {land[7]:.3f} ({land[7]/generic:.1f}x) | at 6: {land[6]:.3f} | at 10: {land[10]:.3f} | at 4: {land[4]:.3f} | at 1: {land[1]:.3f}")

print("\n=== (6) BACKTEST: does a model-vs-market gap that crosses 3 or 7 win more often than an equal gap elsewhere? ===")
print("Model = nfelo closing line (produced pre-game, available 2009-2025). Bet the side nfelo favors vs the market close.")
g = m[m.nfelo_home_line_close.notna() & m.home_line_close.notna()].copy()
g["gap"] = (g.nfelo_home_line_close - g.home_line_close).abs()
lo_ = np.minimum(g.nfelo_home_line_close.abs(), g.home_line_close.abs()); hi_ = np.maximum(g.nfelo_home_line_close.abs(), g.home_line_close.abs())
same_side = np.sign(g.nfelo_home_line_close) == np.sign(g.home_line_close)
g["cross3"] = same_side & (lo_ < 3) & (hi_ > 3)
g["cross7"] = same_side & (lo_ < 7) & (hi_ > 7)
g["cross_key"] = g.cross3 | g.cross7
g["on_key_edge"] = same_side & ((lo_ == 3) | (hi_ == 3) | (lo_ == 7) | (hi_ == 7)) & ~g.cross_key   # gap touches the key number (push risk one side)
g = g[g.gap > 0]
rows = []
for era in ["ALL", "fit<=2021", "test2022-25"]:
    for gb in [(0.5, 0.5), (1.0, 1.0), (1.5, 2.0), (0.5, 1.0), (1.5, 9.0)]:
        for label, mask in [("crosses 3/7", g.cross_key), ("touches 3/7", g.on_key_edge), ("no key involved", ~g.cross_key & ~g.on_key_edge)]:
            h = g if era == "ALL" else g[g.era == era]
            h = h[(h.gap >= gb[0]) & (h.gap <= gb[1]) & mask.loc[h.index]]
            if len(h) == 0: continue
            w, l, p = ats(h.nfelo_home_line_close, h.home_line_close, h.margin)
            rate = w / (w + l) if (w + l) else np.nan
            lo, hi = ci(w, w + l) if (w + l) else (np.nan, np.nan)
            rows.append(dict(era=era, gap=f"{gb[0]}-{gb[1]}", key=label, n=len(h), W=w, L=l, P=p, rate=rate, ci_lo=lo, ci_hi=hi))
t6 = pd.DataFrame(rows); print(t6.round(3).to_string(index=False))
# pooled comparison for gap 0.5-1.0: cross vs no-key
a = t6[(t6.era == "ALL") & (t6.gap == "0.5-1.0") & (t6.key == "crosses 3/7")].iloc[0]
b = t6[(t6.era == "ALL") & (t6.gap == "0.5-1.0") & (t6.key == "no key involved")].iloc[0]
p1, p2 = a.W / (a.W + a.L), b.W / (b.W + b.L); pp = (a.W + b.W) / (a.W + a.L + b.W + b.L)
z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / (a.W + a.L) + 1 / (b.W + b.L)))
print(f"gap 0.5-1.0, crosses-key vs no-key cover rate: {p1:.3f} vs {p2:.3f}, diff {p1-p2:+.3f}, z={z:.2f}, p={2*(1-stats.norm.cdf(abs(z))):.3f}")
t6.to_csv("/home/user/originator-2026-w01/research/experts/market/03_backtest_keycross.csv", index=False)
t3.to_csv("/home/user/originator-2026-w01/research/experts/market/03_landing_probs.csv", index=False)
t4.to_csv("/home/user/originator-2026-w01/research/experts/market/03_cover_table.csv", index=False)
