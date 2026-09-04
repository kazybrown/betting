"""11_open_line_shrink_factor.py - at publication the model only sees the OPEN/current market line.
Part of |model - open| is market movement toward the model, so sigma = sqrt(base^2 + D_open^2)
over-predicts. Fit c in sigma = sqrt(base^2 + (c*D)^2) by Gaussian likelihood on <=2021 for
D = D_open and D = D_base (close; c should be ~1), evaluate 2022-25 (log-score, quintile calibration).
Also: model-vs-OPEN ATS by D_open band with binomial p (a caveat for the market expert).
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/11_open_line_shrink_factor.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, ats

m = build()
m["D_open"] = (m.nfelo_b - m.mkt_open).abs()
def trailing(col, s, k=3):
    t = m[(m.season >= s - k) & (m.season < s)]; return float(np.sqrt((t[col]**2).mean()))
m["base"] = [trailing("e_mkt", s) if s >= 2012 else np.nan for s in m.season]
fit, test = m[(m.era == "fit") & m.base.notna()], m[m.era == "test"]
grid = np.round(np.arange(0, 1.51, 0.05), 2)
for D in ["D_base", "D_open"]:
    nll = [ -np.mean(stats.norm.logpdf(fit.e_nb, 0, np.sqrt(fit.base**2 + (c*fit[D])**2))) for c in grid]
    c_hat = grid[int(np.argmin(nll))]
    # test
    ll = {c: np.mean(stats.norm.logpdf(test.e_nb, 0, np.sqrt(test.base**2 + (c*test[D])**2))) for c in [0, 0.5, c_hat, 1.0]}
    sig = np.sqrt(test.base**2 + (c_hat*test[D])**2)
    q = pd.qcut(sig, 5, labels=False, duplicates="drop"); cal = pd.DataFrame({"s": sig, "e2": test.e_nb**2}).groupby(q).agg(pred=("s", "mean"), real=("e2", lambda x: np.sqrt(x.mean())))
    print(f"{D}: fitted c = {c_hat:.2f} (fit n={len(fit)}) | TEST log-score: c=0 {ll[0]:.5f} | c=0.5 {ll[0.5]:.5f} | c_hat {ll[c_hat]:.5f} | c=1 {ll[1.0]:.5f}")
    print("   quintile calibration with c_hat (pred sigma -> realized RMSE): " + "; ".join(f"{r.pred:.2f}->{r.real:.2f}" for r in cal.itertuples()))
    for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3.0, "MED"), (3.0, 99, "LOW")]:
        s = test[(test[D] >= lo) & (test[D] < hi)]
        print(f"   {name} [{lo},{hi}): n={len(s)} realized rmse_model={np.sqrt((s.e_nb**2).mean()):.2f} | predicted c=1: {np.sqrt(s.base.mean()**2 + (s[D]**2).mean()):.2f} | predicted c_hat: {np.sqrt(s.base.mean()**2 + (c_hat**2)*(s[D]**2).mean()):.2f}")
print("\nmodel (nfelo_b) vs market OPEN, ATS by D_open band (test 2022-25):")
tot = [0, 0]
for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3.0, "MED"), (3.0, 99, "LOW")]:
    s = test[(test.D_open >= lo) & (test.D_open < hi)]; W, L, P = ats(s.nfelo_b, s.mkt_open, s.margin); tot[0] += W; tot[1] += L
    print(f"  {name}: {W}-{L}-{P} ({W/(W+L):.3f}, binom p={stats.binomtest(W, W+L).pvalue:.3f})")
print(f"  overall vs open: {tot[0]}-{tot[1]} ({tot[0]/sum(tot):.3f}, p={stats.binomtest(tot[0], sum(tot)).pvalue:.4f}) | vs close (from 01): 555-555")
print("  -> the open->close move goes toward the model on average; vs the CLOSE the model has no edge. Disagreement measured at publication must be shrunk by c before entering the sigma rule.")
