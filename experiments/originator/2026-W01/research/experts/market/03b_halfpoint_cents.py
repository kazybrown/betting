"""03b_halfpoint_cents.py - half-point value table around 3 and 7 in cover-probability points AND in
price cents (fair-odds equivalent at ~50%: d(price)/dp ~ 400 cents per 1.00 of win probability, so a
mass m_k across integer k is worth ~ 400*m_k cents per full point, ~200*m_k per half point).
Also the exact fair prices for the two half-points around 3 and 7 from the empirical pools. Run from research/.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
m = merged(); m = m[m.mkt_spread.notna()].copy()
m["fav_line"] = m.mkt_spread.abs(); m["fav_margin"] = np.where(m.mkt_spread < 0, m.margin, np.where(m.mkt_spread > 0, -m.margin, m.margin))
rows = []
for k in range(1, 15):
    g = m[(m.fav_line >= k - 2) & (m.fav_line <= k + 2)]; mk = (g.fav_margin == k).mean()
    rows.append(dict(k=k, n=len(g), mass=mk, full_point_cents=400 * mk, half_point_cents=200 * mk))
t = pd.DataFrame(rows); print(t.round(3).to_string(index=False))
gen = t[~t.k.isin([3, 7])].mass.mean()
print(f"\ngeneric integer mass (excl 3,7): {gen:.3f} -> {200*gen:.0f} cents per half point | 3: {200*t.loc[t.k==3,'mass'].item():.0f} c | 7: {200*t.loc[t.k==7,'mass'].item():.0f} c | 6: {200*t.loc[t.k==6,'mass'].item():.0f} c | 10: {200*t.loc[t.k==10,'mass'].item():.0f} c | 14: {200*t.loc[t.k==14,'mass'].item():.0f} c")

def fair_price(pw, pp):
    pl = 1 - pw - pp; s = pw / pl      # stake to win 1 at zero EV
    return -100 * s if s >= 1 else 100 / s
print("\nExact fair prices (favorite side) for a latent fair line at the midpoint, from the empirical two-number pools:")
for x, mks in [(2.75, [2.5, 3.0, 3.5]), (3.25, [2.5, 3.0, 3.5, 4.0]), (6.75, [6.5, 7.0, 7.5]), (7.25, [6.5, 7.0, 7.5, 8.0])]:
    lo, hi = np.floor(x * 2) / 2, np.ceil(x * 2) / 2; s = m[m.fav_line.isin([lo, hi])].fav_margin.values
    out = []
    for mk in mks:
        pw, pp = (s > mk).mean(), (s == mk).mean(); out.append(f"lay {mk}: win {pw:.3f} push {pp:.3f} fair {fair_price(pw, pp):+.0f}")
    print(f"  latent {x} (n={len(s)}): " + " | ".join(out))
