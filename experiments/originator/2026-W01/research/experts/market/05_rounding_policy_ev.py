"""05_rounding_policy_ev.py - Theory 3/4: which rounding rule for an origin spread at x.25/x.75 near
3 or 7 maximizes EV vs the market? EV is computed from EMPIRICAL margin distributions (no parametric
model): a latent fair line at x.75 (or x.25) is represented by the pooled games whose market close was
x.5 or x+1.0 (or x.0 and x.5) in the favorite frame. For each grid market number m near x we compute
P(fav wins / push / loses) directly from that pool, so the EV of any bet is empirical.
Engine rules compared (published number r; bet the side r favors vs m when |r-m| >= T):
  A half-up (spec):          x.25 -> x.5, x.75 -> x+1.0
  B away-from-key, to half:  2.75 -> 2.5, 6.75 -> 6.5 (else = A; note A already sends 3.25->3.5, 7.25->7.5)
  C toward-key:              3.25 -> 3.0, 7.25 -> 7.0 (else = A)
  E probability-space:       no rounding; EV from the landing-mass formula with masses fitted on <=2021 only;
                             bet when formula-EV >= 0.02. Evaluated on the same pools (masses are OOS, pools are not).
  D in-sample ceiling:       bet the side whose EMPIRICAL pool EV >= 0.02 (optimistic upper bound).
Weights over m: continuous gap (market close - nfelo pre-regression line, historic_projected_spreads.csv)
binned to the offsets that put m on the .0/.5 grid. Bootstrap gives CIs on rule-EV differences.
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/05_rounding_policy_ev.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged
rng = np.random.default_rng(7)

m = merged(); m = m[m.mkt_spread.notna()].copy()
m["fav_line"] = m.mkt_spread.abs()
m["fav_margin"] = np.where(m.mkt_spread < 0, m.margin, np.where(m.mkt_spread > 0, -m.margin, m.margin))
m["era"] = np.where(m.season <= 2021, "fit", "test")

# ---- gap weights ----
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h = h.dropna(subset=["home_line_pre_regression", "home_line_close"])
gap = (h.home_line_close - h.home_line_pre_regression).values
print("gap source: historic_projected_spreads.csv seasons", sorted(h.season.unique()), "n =", len(h))
print("market close - nfelo pre-regression line: mean", round(gap.mean(), 3), "SD", round(gap.std(), 3), "| |gap|>=1:", round((np.abs(gap) >= 1).mean(), 3), "| |gap|>=2:", round((np.abs(gap) >= 2).mean(), 3))
OFFS = np.arange(-2.25, 2.26, 0.5)          # offsets that put m on the .0/.5 grid when x is x.25/x.75
w_off = np.array([((gap >= o - 0.25) & (gap < o + 0.25)).mean() for o in OFFS]); w_off = w_off / w_off.sum()
print("weights over market offset (m - x), bins of width 0.5 centred on the grid offsets, |offset|<=2.25:")
print("  ", {float(o): round(float(w), 3) for o, w in zip(OFFS, w_off)})

def pool_for(x, era=None):
    lo, hi = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
    d = m if era is None else m[m.era == era]
    return d[d.fav_line.isin([lo, hi])].fav_margin.values

def outcome_probs(sample, mkt):
    return (sample > mkt).mean(), (sample == mkt).mean(), (sample < mkt).mean()

def ev_unit(pw, pp, pl):        # EV per unit risked at -110
    return (pw - 1.1 * pl) / 1.1

def published(rule, x):
    frac = round(x % 1, 2); base = np.floor(x)
    if frac in (0.0, 0.5): return x
    if rule == "A": return base + 0.5 if frac == 0.25 else base + 1.0
    if rule == "B": return base + 0.5 if x in (2.75, 6.75) else published("A", x)
    if rule == "C": return base if x in (3.25, 7.25) else published("A", x)
    raise ValueError(rule)

# landing masses fitted on <=2021 only (for rule E)
fitp = m[m.era == "fit"]
MASS = {k: ((fitp.fav_margin == k) & (fitp.fav_line >= k - 2) & (fitp.fav_line <= k + 2)).sum() / ((fitp.fav_line >= k - 2) & (fitp.fav_line <= k + 2)).sum() for k in range(1, 25)}
def ev_formula(x, mk):
    """(EV, side) for fair line x (non-integer) vs grid market mk, favorite frame. side +1 = lay fav, -1 = take dog."""
    if mk > x:
        ks = [k for k in range(1, 25) if x < k < mk]; side = -1
    else:
        ks = [k for k in range(1, 25) if mk < k < x]; side = +1
    pw = 0.5 + sum(MASS[k] for k in ks); pp = MASS.get(int(mk), 0.0) if float(mk).is_integer() and mk >= 1 else 0.0
    pl = max(0.0, 1 - pw - pp)
    return ev_unit(pw, pp, pl), side

def rule_ev(rule, x, sample, T, ev_min=0.02):
    tot, freq = 0.0, 0.0
    for o, w in zip(OFFS, w_off):
        mk = round(x + o, 2)
        if mk < 0.5: continue
        pw, pp, pl = outcome_probs(sample, mk)
        ev_fav, ev_dog = ev_unit(pw, pp, pl), ev_unit(pl, pp, pw)
        if rule == "D":
            best = max(ev_fav, ev_dog)
            if best >= ev_min: tot += w * best; freq += w
            continue
        if rule == "E":
            ev, side = ev_formula(x, mk)
            if ev >= ev_min: tot += w * (ev_fav if side > 0 else ev_dog); freq += w
            continue
        r = published(rule, x)
        if r - mk >= T: tot += w * ev_fav; freq += w
        elif mk - r >= T: tot += w * ev_dog; freq += w
    return tot, freq

LATENTS = [2.25, 2.75, 3.25, 3.75, 6.25, 6.75, 7.25, 7.75]
print("\n=== per-cell empirical outcome table (favorite side) for latent x, grid market m ===")
rows = []
for x in LATENTS:
    s = pool_for(x)
    for o in OFFS:
        mk = round(x + o, 2)
        if mk < 0.5: continue
        pw, pp, pl = outcome_probs(s, mk); evf, side = ev_formula(x, mk)
        rows.append(dict(latent=x, pool_n=len(s), market=mk, fav_win=pw, push=pp, dog_win=pl, ev_fav=ev_unit(pw, pp, pl), ev_dog=ev_unit(pl, pp, pw),
                         ev_formula_E=evf, side_E=side, pub_A=published("A", x), pub_B=published("B", x), pub_C=published("C", x)))
cells = pd.DataFrame(rows); print(cells.round(3).to_string(index=False))
cells.to_csv("/home/user/originator-2026-w01/research/experts/market/05_cells.csv", index=False)

print("\n=== weighted EV per game evaluated (per unit risked at -110) of each rule, by latent x and threshold T ===")
res = []
for T in [0.5, 1.0, 1.5]:
    for x in LATENTS:
        s = pool_for(x); r = dict(T=T, latent=x, pool_n=len(s))
        for rule in ["A", "B", "C"]:
            ev, f = rule_ev(rule, x, s, T); r[f"ev_{rule}"] = ev; r[f"bets_{rule}"] = f
        ev, f = rule_ev("E", x, s, T); r["ev_E(prob-space)"] = ev; r["bets_E"] = f
        ev, f = rule_ev("D", x, s, T); r["ev_D(ceiling)"] = ev; r["bets_D"] = f
        res.append(r)
res = pd.DataFrame(res); print(res.round(4).to_string(index=False))
res.to_csv("/home/user/originator-2026-w01/research/experts/market/05_rule_ev.csv", index=False)
print("\nAverage over the 8 latent values (equal weight):")
print(res.groupby("T")[["ev_A", "ev_B", "ev_C", "ev_E(prob-space)", "ev_D(ceiling)"]].mean().round(4).to_string())

print("\n=== Bootstrap (1000 resamples of each pool): EV differences vs A (per game evaluated) ===")
B = 1000
boot_rows = []
for T in [0.5, 1.0, 1.5]:
    for x, alt in [(2.75, "B"), (6.75, "B"), (3.25, "C"), (7.25, "C"), (2.75, "E"), (3.25, "E"), (6.75, "E"), (7.25, "E")]:
        s = pool_for(x); diffs = []
        for _ in range(B):
            bs = rng.choice(s, size=len(s), replace=True)
            diffs.append(rule_ev(alt, x, bs, T)[0] - rule_ev("A", x, bs, T)[0])
        diffs = np.array(diffs); base = rule_ev(alt, x, s, T)[0] - rule_ev("A", x, s, T)[0]
        boot_rows.append(dict(T=T, latent=x, alt=alt, diff=base, ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5), p_gt0=np.mean(diffs > 0), pool_n=len(s)))
bt = pd.DataFrame(boot_rows); print(bt.round(4).to_string(index=False))

print("\n=== Era split: sign of EV(B)-EV(A) at 2.75 / 6.75 and EV(C)-EV(A) at 3.25 / 7.25, fit (<=2021) vs test (2022-25) pools ===")
for T in [0.5, 1.0]:
    for x, alt in [(2.75, "B"), (6.75, "B"), (3.25, "C"), (7.25, "C")]:
        out = []
        for era in ["fit", "test"]:
            s = pool_for(x, era); out.append(f"{era}: n={len(s)} diff={rule_ev(alt, x, s, T)[0]-rule_ev('A', x, s, T)[0]:+.4f}")
        print(f"T={T} latent {x} {alt}-A | " + " | ".join(out))

print("\n=== Landing-mass table (fit <=2021 only, window +/-2) used by rule E; full-sample values for reference ===")
rows = []
for k in range(1, 15):
    g = m[(m.fav_line >= k - 2) & (m.fav_line <= k + 2)]
    rows.append(dict(k=k, mass_fit=MASS[k], mass_full=(g.fav_margin == k).mean(), n_full=len(g)))
mt = pd.DataFrame(rows); print(mt.round(4).to_string(index=False))
mt.to_csv("/home/user/originator-2026-w01/research/experts/market/05_landing_mass_table.csv", index=False)
print("\nEdge formula (favorite frame, fair line x non-integer, grid market m):")
print("  dog side (m > x): P(win)=0.5+sum_{int k, x<k<m} mass_k ; P(push)=mass_m if m integer else 0 ; P(loss)=1-P(win)-P(push)")
print("  fav side (m < x): P(win)=0.5+sum_{int k, m<k<x} mass_k ; same push/loss.  EV/unit at -110 = (P(win)-1.1*P(loss))/1.1")
s = pool_for(2.75)
for mk in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
    evf, side = ev_formula(2.75, mk); pw, pp, pl = outcome_probs(s, mk)
    emp = ev_unit(pw, pp, pl) if side > 0 else ev_unit(pl, pp, pw)
    print(f"  check latent 2.75 vs market {mk}: formula EV={evf:+.3f} ({'fav' if side>0 else 'dog'}) | empirical pool EV same side={emp:+.3f}")
