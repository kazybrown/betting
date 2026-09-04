"""10_robustness_open_line_and_seasons.py - (a) tags are computed at publication, against the market
OPEN/current line, not the close: redo the D-band table and sqrt rule with D_open = |nfelo_b - mkt_open|;
(b) per-season stability of the LOW band's excess error (2022..2025 separately);
(c) sd2-based literal tags per season (is the null stable?).
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/10_robustness_open_line_and_seasons.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, ats

pd.set_option("display.width", 200)
m = build()
m["D_open"] = (m.nfelo_b - m.mkt_open).abs()
m["e_open"] = m.margin + m.mkt_open
fit, test = m[m.era == "fit"], m[m.era == "test"]
print("(a) D vs market OPEN. corr(D_open, D_base) =", round(np.corrcoef(test.D_open, test.D_base)[0, 1], 3), "| mean D_open", round(test.D_open.mean(), 2), "vs D_base", round(test.D_base.mean(), 2))
print("market open RMSE (test) =", round(np.sqrt((test.e_open**2).mean()), 3), "| close RMSE =", round(np.sqrt((test.e_mkt**2).mean()), 3))
def trailing(col, s, k=3):
    t = m[(m.season >= s - k) & (m.season < s)]; return float(np.sqrt((t[col]**2).mean()))
t = test.copy(); t["base"] = [trailing("e_mkt", s) for s in t.season]
sig = np.sqrt(t.base**2 + t.D_open**2)
for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3.0, "MED"), (3.0, 99, "LOW")]:
    s = t[(t.D_open >= lo) & (t.D_open < hi)]; W, L, P = ats(s.nfelo_b, s.mkt_open, s.margin)
    print(f"  {name} D_open in [{lo},{hi}): n={len(s)} share={len(s)/len(t):.2f} rmse_mkt_open={np.sqrt((s.e_open**2).mean()):.2f} rmse_mkt_close={np.sqrt((s.e_mkt**2).mean()):.2f} rmse_model={np.sqrt((s.e_nb**2).mean()):.2f} predicted(sqrt rule)={np.sqrt(s.base.mean()**2 + (s.D_open**2).mean()):.2f} | model vs open ATS {W}-{L}-{P} ({W/(W+L):.3f})")
rho, p = stats.spearmanr(sig, t.e_nb.abs()); print(f"  sqrt rule with D_open: Spearman(sigma, |e_nb|) = {rho:.3f} (p={p:.3f}); coverage k=0.607/1.253/1.615: {[round(float(np.mean(t.e_nb.abs() <= k*sig)),3) for k in (0.607,1.253,1.615)]}")

print("\n(b) per-season LOW band (D_base >= 3) excess RMSE of the model number vs market close, and ATS")
for s in [2022, 2023, 2024, 2025]:
    d = test[test.season == s]; lo = d[d.D_base >= 3]; hi = d[d.D_base < 1.5]
    W, L, P = ats(lo.nfelo_b, lo.mkt, lo.margin)
    print(f"  {s}: LOW n={len(lo)} rmse_model={np.sqrt((lo.e_nb**2).mean()):.2f} rmse_mkt={np.sqrt((lo.e_mkt**2).mean()):.2f} excess={np.sqrt((lo.e_nb**2).mean())-np.sqrt((lo.e_mkt**2).mean()):+.2f} ATS {W}-{L}-{P} | HIGH n={len(hi)} rmse_model={np.sqrt((hi.e_nb**2).mean()):.2f} rmse_mkt={np.sqrt((hi.e_mkt**2).mean()):.2f}")
print("  fit-era seasons (in-sample reference), LOW band excess:")
for s in range(2009, 2022):
    d = fit[fit.season == s]; lo = d[d.D_base >= 3]
    print(f"    {s}: n={len(lo)} excess={np.sqrt((lo.e_nb**2).mean())-np.sqrt((lo.e_mkt**2).mean()):+.2f}", end="")
print()
print("\n(c) per-season MARKET RMSE by D_base band (is the market's error really flat in D every season?)")
for s in range(2016, 2026):
    d = m[m.season == s]; out = []
    for lo, hi in [(0, 1.5), (1.5, 3), (3, 99)]:
        x = d[(d.D_base >= lo) & (d.D_base < hi)]; out.append(f"{np.sqrt((x.e_mkt**2).mean()):.1f}(n={len(x)})")
    print(f"  {s}: " + " | ".join(out))
