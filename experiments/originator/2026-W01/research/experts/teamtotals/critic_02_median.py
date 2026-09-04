"""CRITIC of TT1b (median vs mean). Attacks: bin-boundary sensitivity of the low-tt P(over),
season-clustering, multiple comparisons (8 bins x 2 sides), and the OOS check restricted to the
region where a shade would actually be applied (tt <= 17.5)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from common import load, over_rate, boot_ci

g = load(min_season=1999, verbose=False)
rows = []
for side, sc, tt in [("home", "home_score", "home_tt"), ("away", "away_score", "away_tt")]:
    d = g[["gid", "season", "train", "test", sc, tt]].rename(columns={sc: "score", tt: "tt"}); d["side"] = side; rows.append(d)
L = pd.concat(rows, ignore_index=True); L["resid"] = L.score - L.tt
print("team-games", len(L))
print("\nP(over identity) for low expected team totals, by threshold (pushes excluded):")
print(f"{'thr':>8s} | {'TRAIN 99-21':>26s} | {'TEST 22-25':>26s} | {'ALL 99-25':>26s} | {'2009-25':>26s}")
for thr in [13, 14, 15, 16, 17, 17.5, 20]:
    out = []
    for nm, d in [("tr", L[L.train]), ("te", L[L.test]), ("all", L), ("m", L[L.season >= 2009])]:
        dd = d[d.tt <= thr]; o, n = over_rate(dd.score, dd.tt)
        p = stats.binomtest(int(round(o * n)), n, 0.5).pvalue if n > 0 else np.nan
        out.append(f"{o:.3f} n={n:5d} p={p:.3f}")
    print(f"tt<={thr:>4} | " + " | ".join(f"{s:>26s}" for s in out))
# season clustering: per-season P(over) for tt<=15, count seasons < 0.5
d = L[L.tt <= 15]
per = d.groupby("season").apply(lambda x: over_rate(x.score, x.tt)[0], include_groups=False)
print(f"\ntt<=15: seasons with P(over)<0.5: {(per<0.5).sum()}/{len(per)} (binomial p={stats.binomtest(int((per<0.5).sum()), len(per), 0.5).pvalue:.3f}); season-mean P(over) {per.mean():.3f} (sd {per.std():.3f}, se {per.std()/np.sqrt(len(per)):.3f})")
print("  by season:", per.round(2).to_dict())
# multiple comparisons: 8 bins, min p in train
ebins = [0, 15, 17.5, 20, 22.5, 25, 27.5, 30, 60]
L["ebin"] = pd.cut(L.tt, ebins)
ps = []
for b, d2 in L[L.train].groupby("ebin", observed=True):
    o, n = over_rate(d2.score, d2.tt); ps.append((str(b), n, o, stats.binomtest(int(round(o*n)), n, 0.5).pvalue))
print("\nTRAIN bins: (bin, n, P(over), p):", [(a, b, round(c, 3), round(p, 4)) for a, b, c, p in ps])
print(f"  min p = {min(p for *_, p in ps):.4f}; Bonferroni threshold for 8 bins = {0.05/8:.4f}; Holm-adjusted min p = {min(p for *_, p in ps)*8:.3f}")
# OOS restricted to the region a shade would touch: tt <= 17.5, median-reg line fit on train
tr, te = L[L.train], L[L.test]
q = smf.quantreg("score ~ tt", tr).fit(q=0.5)
lo_te = te[te.tt <= 17.5]
for nm, p in [("median-reg line", q.params["Intercept"] + q.params["tt"] * lo_te.tt), ("flat -0.5", lo_te.tt - 0.5), ("flat -1.0", lo_te.tt - 1.0)]:
    e0 = (lo_te.score - lo_te.tt).abs().values; e1 = (lo_te.score - p).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"OOS 2022-25 tt<=17.5 (n={len(lo_te)}): {nm:>16s} dMAE (identity - rule) {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}]  P(over rule) {over_rate(lo_te.score, p)[0]:.3f}")
# the same rules on the TRAIN low region (in-sample upper bound of what a shade could give)
lo_tr = tr[tr.tt <= 17.5]
for nm, p in [("flat -0.5", lo_tr.tt - 0.5), ("flat -1.0", lo_tr.tt - 1.0)]:
    e0 = (lo_tr.score - lo_tr.tt).abs().values; e1 = (lo_tr.score - p).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"IN-SAMPLE train tt<=17.5 (n={len(lo_tr)}): {nm:>16s} dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}]")
