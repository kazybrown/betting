"""critic_05_T4_base_and_open.py - CRITIC of T4 (sigma rule; SUPPORTED by the expert).
 (1) BASE window horse race, rolling-origin over 2012-2025 (14 seasons): trailing k in {1,2,3,4,5,8}, all
     prior seasons, EWMA(0.3/0.5), evaluated by Gaussian log-score and by |realized RMSE - predicted| of the
     MARKET error (spread and total). Is 'trailing 3' actually best, or a lucky pick?
 (2) how much of the season-to-season RMSE variation is real? between-season variance minus sampling variance;
 (3) the D term vs the OPEN line, decomposed: in-band market RMSE (base noise) vs the cov term; what c does a
     rolling-origin fit give for the open (2016-25), i.e. is the expert's c=0.5 supported by anything?
 (4) k multipliers: empirical vs Gaussian, coverage by season 2022-25 (are they stable?).
 (5) is a 5-band sigma even distinguishable OOS? sigma_sqrt range 12.5-13.6 -> what coverage difference does it make?
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_05_T4_base_and_open.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build

pd.set_option("display.width", 220)
m = build()
m["D_open"] = (m.nfelo_b - m.mkt_open).abs(); m["e_open"] = m.margin + m.mkt_open
seasons = sorted(m.season.unique())
rmse_s = m.groupby("season").apply(lambda d: pd.Series({"spread": np.sqrt((d.e_mkt**2).mean()), "total": np.sqrt((d.e_tot**2).mean()), "n": len(d)}))


def base_pred(col, s, kind, k=3, lam=0.5):
    prior = [x for x in seasons if x < s]
    if kind == "trail":
        use = prior[-k:]
    elif kind == "all":
        use = prior
    elif kind == "ewma":
        w = np.array([lam ** (s - 1 - x) for x in prior]); v = np.array([rmse_s.loc[x, col] ** 2 for x in prior]); return float(np.sqrt(np.sum(w * v) / np.sum(w)))
    d = m[m.season.isin(use)]; return float(np.sqrt((d[col if col != "spread" else "e_mkt"] ** 2).mean())) if col == "spread" else float(np.sqrt((d.e_tot ** 2).mean()))


print("(1) BASE window horse race, rolling-origin 2012-2025 (Gaussian log-score of the MARKET error, mean per game; higher = better)")
specs = [("trail1", "trail", 1), ("trail2", "trail", 2), ("trail3", "trail", 3), ("trail4", "trail", 4), ("trail5", "trail", 5), ("trail8", "trail", 8), ("all_prior", "all", 0), ("ewma0.5", "ewma", 0.5), ("ewma0.7", "ewma", 0.7), ("const_13.3", None, 0)]
for col, ecol in [("spread", "e_mkt"), ("total", "e_tot")]:
    rows = []
    for name, kind, k in specs:
        ll, absdev, per = [], [], []
        for s in range(2012, 2026):
            t = m[m.season == s]
            if kind is None: b = 13.3
            elif kind == "ewma": b = base_pred(col, s, "ewma", lam=k)
            else: b = base_pred(col, s, kind, k=k)
            ll.append(np.mean(stats.norm.logpdf(t[ecol], 0, b))); absdev.append(abs(rmse_s.loc[s, col] - b)); per.append(ll[-1])
        rows.append(dict(spec=name, mean_logscore=np.mean(ll), mean_abs_dev_of_rmse=np.mean(absdev), logscore_2022_25=np.mean(per[-4:])))
    t1 = pd.DataFrame(rows).set_index("spec"); t1["gain_vs_trail3_x1000"] = 1000 * (t1.mean_logscore - t1.loc["trail3", "mean_logscore"])
    print(f"\n  {col.upper()}:"); print(t1.round(4).to_string())
    # paired bootstrap trail3 vs all_prior over per-game log-scores
    lt3, lall = [], []
    for s in range(2012, 2026):
        t = m[m.season == s]; lt3.append(stats.norm.logpdf(t[ecol], 0, base_pred(col, s, "trail", 3))); lall.append(stats.norm.logpdf(t[ecol], 0, base_pred(col, s, "all")))
    dlt = np.concatenate(lt3) - np.concatenate(lall); rng = np.random.default_rng(0); idx = rng.integers(0, len(dlt), (3000, len(dlt))); bs = dlt[idx].mean(1)
    print(f"  trail3 - all_prior per-game log-score: {dlt.mean():+.5f} 95% CI [{np.quantile(bs,.025):+.5f}, {np.quantile(bs,.975):+.5f}] (n={len(dlt)})")

print("\n(2) is season-to-season RMSE drift real? (spread)")
v = rmse_s.spread.values; n_ = rmse_s.n.values; samp_var = np.mean((v ** 2) / (2 * n_))  # var of RMSE estimate ~ sigma^2/(2n)
print(f"  observed SD of season RMSE {v.std(ddof=1):.3f} | sampling SD of a season RMSE (sigma/sqrt(2n)) {np.sqrt(samp_var):.3f} | implied true between-season SD {np.sqrt(max(v.var(ddof=1) - samp_var, 0)):.3f}")
print(f"  seasons: {dict(zip(rmse_s.index, np.round(v, 2)))}")
print(f"  lag-1 autocorrelation of season RMSE: {np.corrcoef(v[:-1], v[1:])[0,1]:+.3f} (persistence is what a trailing window needs)")

print("\n(3) D term vs the OPEN: decomposition and rolling-origin c")
t = m[m.era == "test"].copy()
def trailing(col, s, k=3):
    d = m[(m.season >= s - k) & (m.season < s)]; return float(np.sqrt((d[col] ** 2).mean()))
t["base"] = [trailing("e_mkt", s) for s in t.season]
for D, eref, lab in [("D_base", "e_mkt", "vs CLOSE"), ("D_open", "e_open", "vs OPEN")]:
    print(f"  {lab}:")
    for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3, "MED"), (3, 99, "LOW")]:
        s_ = t[(t[D] >= lo) & (t[D] < hi)]; d = (t.nfelo_b - (t.mkt if D == "D_base" else t.mkt_open))[s_.index]
        v_ref, d2, cov2 = (s_[eref] ** 2).mean(), (d ** 2).mean(), 2 * np.mean(s_[eref] * d)
        print(f"    {name}: n={len(s_)} realized model RMSE {np.sqrt((s_.e_nb**2).mean()):.2f} | expert pred (trailing base {s_.base.mean():.2f}) {np.sqrt(s_.base.mean()**2 + d2):.2f} | in-band ref-line RMSE {np.sqrt(v_ref):.2f} -> identity {np.sqrt(v_ref + d2):.2f} | 2cov {cov2:+.2f} (w_implied {-0.5*cov2/d2:+.2f})")
print("  rolling-origin c (fit Gaussian c on all prior seasons >= 2012, evaluate season s) for D_open and D_base:")
grid = np.round(np.arange(0, 1.51, 0.05), 2)
m["base"] = [trailing("e_mkt", s) if s >= 2012 else np.nan for s in m.season]
for D in ["D_base", "D_open"]:
    rows = []
    for s in range(2016, 2026):
        f = m[(m.season < s) & m.base.notna()]; tt = m[m.season == s]
        nll = [-np.mean(stats.norm.logpdf(f.e_nb, 0, np.sqrt(f.base ** 2 + (c * f[D]) ** 2))) for c in grid]; c_hat = grid[int(np.argmin(nll))]
        ll = {c: np.mean(stats.norm.logpdf(tt.e_nb, 0, np.sqrt(tt.base ** 2 + (c * tt[D]) ** 2))) for c in [0, 0.5, 1.0]}
        rows.append(dict(season=s, c_hat=c_hat, gain_c1_vs_c0_x1000=1000 * (ll[1.0] - ll[0]), gain_c05_vs_c0_x1000=1000 * (ll[0.5] - ll[0])))
    r = pd.DataFrame(rows).set_index("season"); print(f"    {D}: c_hat by year {r.c_hat.to_dict()} | mean gain c=1 vs 0: {r.gain_c1_vs_c0_x1000.mean():+.3f} (x1000), positive in {(r.gain_c1_vs_c0_x1000>0).sum()}/10 | c=0.5 vs 0: {r.gain_c05_vs_c0_x1000.mean():+.3f}, positive in {(r.gain_c05_vs_c0_x1000>0).sum()}/10")

print("\n(4) k multipliers: coverage of the sqrt rule (trailing base, D_base, k=0.607/1.253/1.615) by test season, spread; Gaussian k for comparison")
for s in [2022, 2023, 2024, 2025]:
    tt = t[t.season == s]; sig = np.sqrt(tt.base ** 2 + tt.D_base ** 2); ae = tt.e_nb.abs()
    emp = [float(np.mean(ae <= k * sig)) for k in (0.607, 1.253, 1.615)]; gau = [float(np.mean(ae <= k * sig)) for k in (0.674, 1.282, 1.645)]
    print(f"  {s}: n={len(tt)} empirical-k coverage {np.round(emp,3)} | Gaussian-k coverage {np.round(gau,3)} (targets .5/.8/.9)")
sig = np.sqrt(t.base ** 2 + t.D_base ** 2); ae = t.e_nb.abs()
print(f"  ALL: empirical {[round(float(np.mean(ae <= k*sig)),3) for k in (0.607,1.253,1.615)]} | Gaussian {[round(float(np.mean(ae <= k*sig)),3) for k in (0.674,1.282,1.645)]} | median |e|/sigma = {np.median(ae/sig):.3f}")
print("  -> is the p50 multiplier 0.607 vs 0.674 a key-number effect? share of |e_nb| <= 3: %.3f, share of |margin| in {3,7}: %.3f" % (float(np.mean(ae <= 3)), float(np.mean(t.margin.abs().isin([3, 7])))))

print("\n(5) does the per-game sigma matter at all OOS? PIT-based check: sqrt rule vs base-only, share of games whose interval verdict changes")
sig0 = t.base; sig1 = np.sqrt(t.base ** 2 + t.D_base ** 2)
for k, p in [(1.253, 0.8), (1.615, 0.9)]:
    in0, in1 = ae <= k * sig0, ae <= k * sig1; print(f"  {int(p*100)}% interval: base-only coverage {in0.mean():.3f}, sqrt-rule {in1.mean():.3f}; verdict changes in {int((in0 != in1).sum())} of {len(t)} games ({100*(in0!=in1).mean():.1f}%)")
print(f"  sigma_sqrt / base: mean {float((sig1/sig0).mean()):.4f}, p90 {float((sig1/sig0).quantile(.9)):.4f}, max {float((sig1/sig0).max()):.3f}")
