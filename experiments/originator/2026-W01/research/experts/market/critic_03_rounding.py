"""critic_03_rounding.py - adversarial re-analysis of T3 (key-number-aware rounding of x.25/x.75 origins).
Attacks: (a) reproduce the pooled-EV rule comparison (expert's 05 method, identical code path);
(b) rolling 3-block stability of B-A (2.75, 6.75) and C-A (3.25, 7.25); (c) placebo: the same 'round x.75 down
to the half' rule at NON-key x.75 (1.75, 4.75, 5.75, 8.75) and 'round x.25 down to the integer' at non-key x.25;
(d) actual closing juice by number instead of a flat -110 (vig-free two-way prices, by era because the odds
source changes in 2023); (e) the expert's own fair-price argument applied symmetrically at 3.25 and 7.25;
(f) multiplicity of the bootstrap 'P>0=0.99' cell.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/critic_03_rounding.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
rng = np.random.default_rng(7)
m = merged(); m = m[m.mkt_spread.notna()].copy()
m["fav_line"] = m.mkt_spread.abs()
m["fav_margin"] = np.where(m.mkt_spread < 0, m.margin, np.where(m.mkt_spread > 0, -m.margin, m.margin))
m["fav_odds"] = np.where(m.mkt_spread < 0, m.home_spread_odds, m.away_spread_odds)
m["dog_odds"] = np.where(m.mkt_spread < 0, m.away_spread_odds, m.home_spread_odds)
m["block"] = pd.cut(m.season, [2008, 2014, 2021, 2025], labels=["2009-14", "2015-21", "2022-25"])
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False).dropna(subset=["home_line_pre_regression", "home_line_close"])
gap = (h.home_line_close - h.home_line_pre_regression).values
OFFS = np.arange(-2.25, 2.26, 0.5)
w_off = np.array([((gap >= o - 0.25) & (gap < o + 0.25)).mean() for o in OFFS]); w_off /= w_off.sum()

def pool_for(x, df=None):
    df = m if df is None else df; lo, hi = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
    return df[df.fav_line.isin([lo, hi])].fav_margin.values
def probs(s, mk): return (s > mk).mean(), (s == mk).mean(), (s < mk).mean()
def ev110(pw, pp, pl): return (pw - 1.1 * pl) / 1.1
def pub(rule, x, keys=(3, 7)):
    frac = round(x % 1, 2); base = np.floor(x)
    if frac in (0.0, 0.5): return x
    A = base + 0.5 if frac == 0.25 else base + 1.0
    if rule == "A": return A
    if rule == "B": return base + 0.5 if (frac == 0.75 and base + 1 in keys) else A
    if rule == "C": return base if (frac == 0.25 and base in keys) else A
def rule_ev(rule, x, s, T, keys=(3, 7), price=None):
    tot = 0.0
    for o, w in zip(OFFS, w_off):
        mk = round(x + o, 2)
        if mk < 0.5: continue
        pw, pp, pl = probs(s, mk); r = pub(rule, x, keys)
        if price is None: ef, ed = ev110(pw, pp, pl), ev110(pl, pp, pw)
        else:
            df_, dd_ = price(mk); ef = pw * (df_ - 1) - pl; ed = pl * (dd_ - 1) - pw
        if r - mk >= T: tot += w * ef
        elif mk - r >= T: tot += w * ed
    return tot

print("=== (a) reproduce: B-A at 2.75 / 6.75, C-A at 3.25 / 7.25 (pooled 2009-25, -110 flat) ===")
for T in [0.5, 1.0, 1.5]:
    print(f"T={T}: " + " | ".join(f"{x} {alt}-A {rule_ev(alt, x, pool_for(x), T) - rule_ev('A', x, pool_for(x), T):+.4f}" for x, alt in [(2.75, 'B'), (6.75, 'B'), (3.25, 'C'), (7.25, 'C')]))

print("\n=== (b) rolling blocks ===")
for T in [0.5, 1.0]:
    for x, alt in [(2.75, "B"), (6.75, "B"), (3.25, "C"), (7.25, "C")]:
        out = []
        for blk, g in m.groupby("block", observed=True):
            s = pool_for(x, g); out.append(f"{blk}: n={len(s)} {rule_ev(alt, x, s, T) - rule_ev('A', x, s, T):+.4f}")
        print(f"T={T} latent {x} {alt}-A | " + " | ".join(out))

print("\n=== (c) placebo: same rules at non-key numbers (treat that integer as the 'key') ===")
for T in [0.5, 1.0]:
    row = []
    for x in [1.75, 2.75, 4.75, 5.75, 6.75, 8.75, 9.75]:
        k = int(np.floor(x)) + 1; s = pool_for(x); row.append(f"{x}->{x-0.25} {rule_ev('B', x, s, T, keys=(k,)) - rule_ev('A', x, s, T, keys=(k,)):+.4f} (n={len(s)})")
    print(f"T={T} round-down-to-half rule: " + " | ".join(row))
    row = []
    for x in [2.25, 3.25, 4.25, 5.25, 7.25, 8.25, 9.25]:
        k = int(np.floor(x)); s = pool_for(x); row.append(f"{x}->{k} {rule_ev('C', x, s, T, keys=(k,)) - rule_ev('A', x, s, T, keys=(k,)):+.4f} (n={len(s)})")
    print(f"T={T} round-down-to-integer rule: " + " | ".join(row))

print("\n=== (d) actual closing juice by number (favorite frame), vig-free, by odds-source era ===")
def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.maximum(np.abs(o), 100))
j = m[m.fav_odds.notna()].copy(); j["era"] = np.where(j.season <= 2022, "2009-22 (low-vig book)", "2023-25 (-110 book)")
j["p_fav_imp"] = (1 / dec(j.fav_odds)) / (1 / dec(j.fav_odds) + 1 / dec(j.dog_odds))
tab = j[j.fav_line.isin([1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 6.5, 7, 7.5, 8, 10])].groupby(["era", "fav_line"]).agg(n=("p_fav_imp", "size"), fav_odds_med=("fav_odds", "median"), dog_odds_med=("dog_odds", "median"), p_fav_imp=("p_fav_imp", "mean"), fav_cover=("fav_margin", lambda s: np.nan))
for (era, k), r in tab.iterrows():
    g = j[(j.era == era) & (j.fav_line == k)]; tab.loc[(era, k), "fav_cover"] = (g.fav_margin > k).sum() / max((g.fav_margin != k).sum(), 1)
print(tab.round(3).to_string())
# EV of the key cells using vig-free actual prices: use decimal odds = 1/p_imp_side * (1 - overround share) -> simply use the median actual prices per number (pooled)
med = j.groupby("fav_line").agg(f=("fav_odds", "median"), d=("dog_odds", "median"))
def price(mk):
    if mk in med.index: return float(dec(med.loc[mk, "f"])), float(dec(med.loc[mk, "d"]))
    return 1 / 1.1 + 1, 1 / 1.1 + 1
print("\nRule EV with median ACTUAL prices per number instead of flat -110 (pooled 2009-25):")
for T in [0.5, 1.0]:
    print(f"T={T}: " + " | ".join(f"{x} {alt}-A {rule_ev(alt, x, pool_for(x), T, price=price) - rule_ev('A', x, pool_for(x), T, price=price):+.4f}" for x, alt in [(2.75, 'B'), (6.75, 'B'), (3.25, 'C'), (7.25, 'C')]))
s = pool_for(2.75)
for mk in [2.0, 2.5, 3.0, 3.5]:
    pw, pp, pl = probs(s, mk); df_, dd_ = price(mk)
    print(f"  latent 2.75 vs market {mk}: fav EV -110 {ev110(pw, pp, pl):+.3f} / actual {pw*(df_-1)-pl:+.3f} | dog EV -110 {ev110(pl, pp, pw):+.3f} / actual {pl*(dd_-1)-pw:+.3f}  (median prices fav {med.loc[mk,'f']:.0f} dog {med.loc[mk,'d']:.0f})")

print("\n=== (e) expert's fair-price argument applied symmetrically ===")
def fair(pw, pp):
    pl = 1 - pw - pp; s_ = pw / pl; return -100 * s_ if s_ >= 1 else 100 / s_
for x, cands in [(2.75, [2.5, 3.0]), (3.25, [3.0, 3.5]), (6.75, [6.5, 7.0]), (7.25, [7.0, 7.5])]:
    s = pool_for(x); out = []
    for mk in cands:
        pw, pp, pl = probs(s, mk); out.append(f"lay {mk}: fair {fair(pw, pp):+.0f} (|dist from -110| = {abs(fair(pw, pp) + 110):.0f}c)")
    print(f"latent {x}: " + " | ".join(out) + f"  -> nearest-to-fair publication = {cands[int(np.argmin([abs(fair(probs(s, c)[0], probs(s, c)[1]) + 110) for c in cands]))]}; half-up publishes {pub('A', x)}, rule B publishes {pub('B', x)}, rule C publishes {pub('C', x)}")

print("\n=== (f) multiplicity: PAIRED bootstrap P(>0) for all 8 latents x 3 thresholds under rule B/C (24 cells) + 4 generic placebo latents ===")
B = 400; cells = []
for T in [0.5, 1.0, 1.5]:
    for x in [2.25, 2.75, 3.25, 3.75, 6.25, 6.75, 7.25, 7.75, 5.75, 8.75, 9.75, 4.25]:
        alt = "B" if round(x % 1, 2) == 0.75 else "C"; keys = (int(np.floor(x)) + 1,) if alt == "B" else (int(np.floor(x)),)
        s = pool_for(x); base = rule_ev(alt, x, s, T, keys) - rule_ev("A", x, s, T, keys)
        diffs = []
        for _ in range(B):
            bs = rng.choice(s, len(s), replace=True); diffs.append(rule_ev(alt, x, bs, T, keys) - rule_ev("A", x, bs, T, keys))   # PAIRED resample (same as expert)
        diffs = np.array(diffs)
        cells.append(dict(T=T, latent=x, rule=alt, diff=base, p_gt0=(diffs > 0).mean(), key=(x in (2.75, 3.25, 6.75, 7.25))))
c = pd.DataFrame(cells); print(c.round(4).to_string(index=False))
print("cells with P(>0)>=0.975:", int((c.p_gt0 >= 0.975).sum()), "of", len(c), "| which:", c[c.p_gt0 >= 0.975][["T","latent","rule"]].values.tolist(), "| cells with P(>0)<=0.025:", int((c.p_gt0 <= 0.025).sum()))
