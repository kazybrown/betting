"""04b_totals_rounding_ev.py - Theory 5: does rounding a total near 41/44/47/51 matter? Same pooled-EV
method as 05: latent total at x.25/x.75 = pool of games with the two adjacent grid closing totals; grid
market totals at offsets; rules A (half-up) vs K (away from key total, to the half: 43.75->43.5, 44.25->44.5,
etc., i.e. never publish the key integer from a quarter) vs K2 (toward key: 43.75->44, 44.25->44).
Bet 'over' if published > market by >= T, 'under' if < by >= T. Run from research/.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
rng = np.random.default_rng(3)
m = merged(); m = m[m.mkt_total.notna()].copy()
OFFS = np.arange(-2.25, 2.26, 0.5)
# gap weights: use the same spread-gap shape as 05 (no originator total history); symmetric bins
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False).dropna(subset=["home_line_pre_regression", "home_line_close"])
gap = (h.home_line_close - h.home_line_pre_regression).values
w_off = np.array([((gap >= o - 0.25) & (gap < o + 0.25)).mean() for o in OFFS]); w_off /= w_off.sum()
KEYS = [41, 44, 47, 51]
def pool(x):
    lo, hi = np.floor(x * 2) / 2, np.ceil(x * 2) / 2; return m[m.mkt_total.isin([lo, hi])].total_pts.values
def probs(s, mk): return (s > mk).mean(), (s == mk).mean(), (s < mk).mean()
def ev(pw, pp, pl): return (pw - 1.1 * pl) / 1.1
def pub(rule, x):
    frac = round(x % 1, 2); base = np.floor(x)
    if rule == "A": return base + 0.5 if frac == 0.25 else base + 1.0
    if rule == "K":   # away from key, to the half
        if frac == 0.75 and base + 1 in KEYS: return base + 0.5
        return pub("A", x)
    if rule == "K2":  # toward key
        if frac == 0.25 and base in KEYS: return base
        return pub("A", x)
def rule_ev(rule, x, s, T):
    tot, f = 0.0, 0.0
    for o, w in zip(OFFS, w_off):
        mk = round(x + o, 2); po, pp, pu = probs(s, mk); r = pub(rule, x)
        if r - mk >= T: tot += w * ev(po, pp, pu); f += w
        elif mk - r >= T: tot += w * ev(pu, pp, po); f += w
    return tot, f
rows = []
for T in [0.5, 1.0]:
    for k in KEYS:
        for x in [k - 0.25, k + 0.25]:
            s = pool(x); r = dict(T=T, latent=x, pool_n=len(s))
            for rule in ["A", "K", "K2"]:
                e, f = rule_ev(rule, x, s, T); r[f"ev_{rule}"] = e; r[f"bets_{rule}"] = f
            alt = "K" if x % 1 == 0.75 else "K2"
            diffs = []
            for _ in range(500):
                bs = rng.choice(s, len(s), replace=True); diffs.append(rule_ev(alt, x, bs, T)[0] - rule_ev("A", x, bs, T)[0])
            r["alt"] = alt; r["diff_alt_minus_A"] = r[f"ev_{alt}"] - r["ev_A"]; r["ci_lo"] = np.percentile(diffs, 2.5); r["ci_hi"] = np.percentile(diffs, 97.5)
            rows.append(r)
t = pd.DataFrame(rows); print(t.round(4).to_string(index=False))
print("\nmean diff (alt - A) over the 8 key-adjacent latents:", t.groupby("T").diff_alt_minus_A.mean().round(4).to_dict())
print("mean |diff|:", t.groupby("T").diff_alt_minus_A.apply(lambda s: s.abs().mean()).round(4).to_dict())
print("\nFor scale, the same comparison at NON-key totals 43 / 46 / 49:")
for T in [0.5, 1.0]:
    for k in [43, 46, 49]:
        for x in [k - 0.25, k + 0.25]:
            s = pool(x); alt = "K" if x % 1 == 0.75 else "K2"
            KEYS_SAVE = KEYS[:]; KEYS[:] = [k]
            d = rule_ev(alt, x, s, T)[0] - rule_ev("A", x, s, T)[0]; KEYS[:] = KEYS_SAVE
            print(f"  T={T} latent {x} (n={len(s)}): treating {k} as key, diff = {d:+.4f}")
