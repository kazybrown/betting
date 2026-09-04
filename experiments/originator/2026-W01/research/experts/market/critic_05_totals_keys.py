"""critic_05_totals_keys.py - adversarial re-analysis of T5 (totals key numbers 41/44/47/51).
Attacks: (a) reproduce masses and push rates; (b) 3-block rolling stability of the 'key' vs 'other' mass ratio;
(c) full placebo: run the rounding-rule EV comparison treating EVERY integer 37-54 as the 'key' and rank the
true keys inside that distribution; (d) juice on integer total lines (is the .0 vs .5 choice already priced?).
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_05_totals_keys.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
m = merged(); m = m[m.mkt_total.notna()].copy()
m["block"] = pd.cut(m.season, [2008, 2014, 2021, 2025], labels=["2009-14", "2015-21", "2022-25"])
KEYS = [41, 44, 47, 51]
def mass(df, k, w=3):
    g = df[(df.mkt_total >= k - w) & (df.mkt_total <= k + w)]; return (g.total_pts == k).mean(), len(g)
print("=== (a) reproduce landing masses (window +/-3) ===")
km = np.mean([mass(m, k)[0] for k in KEYS]); om = np.mean([mass(m, k)[0] for k in range(34, 58) if k not in KEYS])
print(f"key mean {km:.4f} | other mean {om:.4f} | ratio {km/om:.2f}")
print("\n=== (b) rolling blocks: key vs other mass ===")
for blk, g in m.groupby("block", observed=True):
    kmb = np.mean([mass(g, k)[0] for k in KEYS]); omb = np.mean([mass(g, k)[0] for k in range(36, 56) if k not in KEYS])
    print(f"{blk}: n={len(g)} key {kmb:.4f} other {omb:.4f} ratio {kmb/omb:.2f} | per key: " + ", ".join(f"{k}:{mass(g,k)[0]:.3f}" for k in KEYS))
# is the ordering stable? rank of each integer's mass within block
ranks = {}
for blk, g in m.groupby("block", observed=True):
    s = pd.Series({k: mass(g, k)[0] for k in range(37, 55)}); ranks[blk] = s.rank(ascending=False).astype(int)
print("rank of mass (1 = heaviest) by block:\n", pd.DataFrame(ranks).loc[[40, 41, 43, 44, 47, 48, 51, 55] if False else [37, 40, 41, 43, 44, 45, 47, 48, 51]].to_string())

print("\n=== (c) full placebo of the rounding-rule EV comparison (expert's 04b method) for every integer 37-54 ===")
rng = np.random.default_rng(5)
OFFS = np.arange(-2.25, 2.26, 0.5)
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False).dropna(subset=["home_line_pre_regression", "home_line_close"])
gap = (h.home_line_close - h.home_line_pre_regression).values
w_off = np.array([((gap >= o - 0.25) & (gap < o + 0.25)).mean() for o in OFFS]); w_off /= w_off.sum()
def pool(x):
    lo, hi = np.floor(x * 2) / 2, np.ceil(x * 2) / 2; return m[m.mkt_total.isin([lo, hi])].total_pts.values
def probs(s, mk): return (s > mk).mean(), (s == mk).mean(), (s < mk).mean()
def ev(pw, pp, pl): return (pw - 1.1 * pl) / 1.1
def pub(rule, x, key):
    frac = round(x % 1, 2); base = np.floor(x)
    A = base + 0.5 if frac == 0.25 else base + 1.0
    if rule == "A": return A
    if rule == "K": return base + 0.5 if (frac == 0.75 and base + 1 == key) else A
    if rule == "K2": return base if (frac == 0.25 and base == key) else A
def rule_ev(rule, x, s, T, key):
    tot = 0.0
    for o, w in zip(OFFS, w_off):
        mk = round(x + o, 2); po, pp, pu = probs(s, mk); r = pub(rule, x, key)
        if r - mk >= T: tot += w * ev(po, pp, pu)
        elif mk - r >= T: tot += w * ev(pu, pp, po)
    return tot
rows = []
for k in range(37, 55):
    for T in [0.5, 1.0]:
        d1 = rule_ev("K", k - 0.25, pool(k - 0.25), T, k) - rule_ev("A", k - 0.25, pool(k - 0.25), T, k)
        d2 = rule_ev("K2", k + 0.25, pool(k + 0.25), T, k) - rule_ev("A", k + 0.25, pool(k + 0.25), T, k)
        rows.append(dict(k=k, iskey=k in KEYS, thr=T, n_lo=len(pool(k - 0.25)), n_hi=len(pool(k + 0.25)), diff_K=d1, diff_K2=d2, mean_diff=(d1 + d2) / 2))
t = pd.DataFrame(rows)
for T in [0.5, 1.0]:
    s = t[t.thr == T].sort_values("mean_diff", ascending=False)
    print(f"T={T}: mean diff (alt-A) at keys {s[s.iskey].mean_diff.mean():+.4f} vs non-keys {s[~s.iskey].mean_diff.mean():+.4f} | SD across non-keys {s[~s.iskey].mean_diff.std():.4f}")
    print("   ranking (1=largest gain from key-aware rounding): " + ", ".join(f"{int(r.k)}{'*' if r.iskey else ''}:{r.mean_diff:+.3f}" for r in s.itertuples()))
    print(f"   Mann-Whitney keys vs non-keys p = {stats.mannwhitneyu(s[s.iskey].mean_diff, s[~s.iskey].mean_diff).pvalue:.3f}")

print("\n=== (d) juice at integer total lines (over vs under odds) ===")
def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.maximum(np.abs(o), 100))
j = m[m.over_odds.notna()].copy(); j["p_over"] = (1 / dec(j.over_odds)) / (1 / dec(j.over_odds) + 1 / dec(j.under_odds))
g = j[j.mkt_total.isin(range(36, 56))].groupby("mkt_total").agg(n=("p_over", "size"), over_odds=("over_odds", "median"), under_odds=("under_odds", "median"), p_over_imp=("p_over", "mean"), over_real=("total_err_mkt", lambda s: (s > 0).sum() / max((s != 0).sum(), 1)))
g["key"] = g.index.isin(KEYS); print(g.round(3).to_string())
print("mean implied over prob at key integer lines:", round(g[g.key].p_over_imp.mean(), 4), "| other integers:", round(g[~g.key].p_over_imp.mean(), 4), "| half-point lines:", round(j[(j.mkt_total % 1) == 0.5].p_over.mean(), 4))
