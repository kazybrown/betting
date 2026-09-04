"""04_totals_key_numbers.py - Theory 5: totals key numbers (41, 44, 47, 51). Are they modal enough
to change the rounding rule? Landing probabilities, push probability at the line, and half-point
value versus a generic integer.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/04_totals_key_numbers.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged

m = merged()
m = m[m.mkt_total.notna()].copy()
m["era"] = np.where(m.season <= 2021, "fit<=2021", "test2022-25")
def ci(k, n):
    r = stats.binomtest(int(k), int(n), 0.5).proportion_ci(method="wilson"); return r.low, r.high

print("n games:", len(m), "| total_pts mean", round(m.total_pts.mean(), 2), "| median", m.total_pts.median())
print("\n=== (1) Unconditional distribution of total points, 30..60 ===")
vc = m.total_pts.value_counts(normalize=True)
rows = [dict(total=k, share=vc.get(k, 0.0), n=int((m.total_pts == k).sum())) for k in range(30, 61)]
t1 = pd.DataFrame(rows)
t1["share_ma5"] = t1.share.rolling(5, center=True).mean()     # local smoothed baseline
t1["excess_vs_local"] = t1.share - t1.share_ma5
print(t1.round(4).to_string(index=False))
print("top-10 most common totals:", m.total_pts.value_counts().head(10).to_dict())
print("share of ODD totals:", round((m.total_pts % 2 == 1).mean(), 3), "(structural: odd totals need a FG/safety/2pt imbalance)")

print("\n=== (2) Landing probability near the line: P(total == k | line within +/-3 of k) ===")
rows = []
for k in range(34, 58):
    g = m[(m.mkt_total >= k - 3) & (m.mkt_total <= k + 3)]
    if len(g) < 50: continue
    c = int((g.total_pts == k).sum()); lo, hi = ci(c, len(g))
    rows.append(dict(k=k, n=len(g), land_prob=c / len(g), ci_lo=lo, ci_hi=hi,
                     land_fit=(g[g.era == "fit<=2021"].total_pts == k).mean(), land_test=(g[g.era == "test2022-25"].total_pts == k).mean()))
t2 = pd.DataFrame(rows); print(t2.round(3).to_string(index=False))
keys = [41, 44, 47, 51]
kmass = t2[t2.k.isin(keys)].land_prob.mean(); omass = t2[~t2.k.isin(keys)].land_prob.mean()
print(f"mean landing mass at 'key' totals {keys}: {kmass:.4f} | mean at other integers 34-57: {omass:.4f} | ratio {kmass/omass:.2f}")
print("largest landing masses:", t2.sort_values("land_prob", ascending=False).head(6)[["k", "land_prob", "n"]].to_string(index=False))

print("\n=== (3) Push probability at the exact line: P(total == line | line == k), integer lines ===")
rows = []
for k in range(36, 56):
    g = m[m.mkt_total == k]
    if len(g) < 30: continue
    c = int((g.total_pts == k).sum()); lo, hi = ci(c, len(g))
    rows.append(dict(line=k, n=len(g), push=c / len(g), ci_lo=lo, ci_hi=hi, over=(g.total_pts > k).mean(), under=(g.total_pts < k).mean()))
t3 = pd.DataFrame(rows); print(t3.round(3).to_string(index=False))
print("pooled push prob, integer total lines 36-55:", round(t3.pushes.sum() / t3.n.sum(), 4) if "pushes" in t3 else round((t3.push * t3.n).sum() / t3.n.sum(), 4))
print("pooled push prob at 'key' totals:", round((t3[t3.line.isin(keys)].push * t3[t3.line.isin(keys)].n).sum() / t3[t3.line.isin(keys)].n.sum(), 4),
      "| at other integer lines:", round((t3[~t3.line.isin(keys)].push * t3[~t3.line.isin(keys)].n).sum() / t3[~t3.line.isin(keys)].n.sum(), 4))

print("\n=== (4) Comparison with spreads: half-point value (cover-prob points) ===")
sp = m[m.mkt_spread.notna()].copy(); sp["fav_line"] = sp.mkt_spread.abs()
sp["fav_margin"] = np.where(sp.mkt_spread < 0, sp.margin, -sp.margin)
def land_sp(k):
    g = sp[(sp.fav_line >= k - 2) & (sp.fav_line <= k + 2)]; return (g.fav_margin == k).mean()
print(f"spread mass at 3: {land_sp(3):.3f} | at 7: {land_sp(7):.3f} | generic spread integer (4,5,8,9): {np.mean([land_sp(k) for k in [4,5,8,9]]):.3f}")
print(f"totals mass at key totals: {kmass:.3f} | other totals: {omass:.3f}")
print("=> extra half-point value at a 'key' total vs a generic total integer =", round(kmass - omass, 4), "cover-prob points")
t2.to_csv("/home/user/originator-2026-w01/research/experts/market/04_totals_landing.csv", index=False)
