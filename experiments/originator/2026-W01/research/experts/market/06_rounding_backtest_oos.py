"""06_rounding_backtest_oos.py - out-of-sample check of the rounding rules with a genuinely continuous
origin number. Model: OLS margin ~ elo_dif_pts + hfa_pts + qb adj (fit on 2009-2021), continuous spread
pred = -fitted margin. Compare rules A/B/C (see 05) on 2022-2025: bets triggered at |published - market| >= T.
Because the rules only differ at x.75/x.25 adjacent to 3 and 7, the number of games where the DECISION
differs is small -> expect INCONCLUSIVE; reported for honesty. Also reports the probability-space rule D
using the landing-mass table fitted on <=2021 only (so D is out-of-sample here).
Run: cd /home/user/originator-2026-w01/research && python3 experts/market/06_rounding_backtest_oos.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import merged

m = merged(); m = m[m.mkt_spread.notna() & m.elo_dif_pts.notna()].copy()
m["qb_adj_pts"] = (m.home_538_qb_adj.fillna(0) - m.away_538_qb_adj.fillna(0)) / 25.0
fit = m[m.season <= 2021]; test = m[m.season >= 2022]
X = sm.add_constant(fit[["elo_dif_pts", "hfa_pts", "qb_adj_pts"]]); ols = sm.OLS(fit.margin, X).fit()
print("OLS fit <=2021 (margin ~ elo_dif + hfa + qb):", ols.params.round(3).to_dict(), "| n =", len(fit))
Xt = sm.add_constant(test[["elo_dif_pts", "hfa_pts", "qb_adj_pts"]])
test = test.copy(); test["pred_margin"] = ols.predict(Xt); test["origin"] = -test.pred_margin     # ORIGINATOR convention
print("test 2022-25: n =", len(test), "| MAE origin", round((test.margin + test.origin).abs().mean(), 3), "| MAE market", round((test.margin + test.mkt_spread).abs().mean(), 3))
print("SD(origin - market) =", round((test.origin - test.mkt_spread).std(), 3))

# favorite-frame representation using the ORIGIN's favorite (sign of origin)
sgn = np.where(test.origin < 0, 1, -1)                  # +1 if origin favors home
test["x"] = test.origin.abs()                           # origin fair line, favorite frame
test["mk"] = -sgn * test.mkt_spread                     # market line in the same frame (can be negative if market favors the other side)
test["fm"] = sgn * test.margin                          # origin-favorite's margin
q = (test.x * 4).round() / 4                            # quarter-point resolution of the origin
test["xq"] = q

def pub(rule, x):
    frac = round(x % 1, 2); base = np.floor(x)
    if frac in (0.0, 0.5): return x
    if rule == "A": return base + 0.5 if frac == 0.25 else base + 1.0
    if rule == "B": return base + 0.5 if x in (2.75, 6.75) else pub("A", x)
    if rule == "C": return base if x in (3.25, 7.25) else pub("A", x)

# landing mass fitted on <=2021 only
fit2 = fit.copy(); fit2["fav_line"] = fit2.mkt_spread.abs(); fit2["fav_margin"] = np.where(fit2.mkt_spread < 0, fit2.margin, -fit2.margin)
mass = {k: ((fit2.fav_margin == k) & (fit2.fav_line >= k - 2) & (fit2.fav_line <= k + 2)).sum() / ((fit2.fav_line >= k - 2) & (fit2.fav_line <= k + 2)).sum() for k in range(1, 22)}
mass_neg = {k: ((fit2.fav_margin == -k) & (fit2.fav_line >= 0) & (fit2.fav_line <= 3)).sum() / ((fit2.fav_line >= 0) & (fit2.fav_line <= 3)).sum() for k in range(1, 22)}
def mass_at(k):
    return mass.get(k, 0.0) if k > 0 else (mass_neg.get(-k, 0.0) if k < 0 else 0.003)

def ev_prob_space(x, mk):
    """EV (per unit at -110) of best side for fair line x vs market mk, both in origin-favorite frame."""
    if mk > x:   # dog has value
        ks = [k for k in range(int(np.floor(x)) + 1, int(np.ceil(mk))) if k > x and k < mk]
        pw = 0.5 + sum(mass_at(k) for k in ks); pp = mass_at(int(mk)) if float(mk).is_integer() else 0.0
        side = -1
    else:
        ks = [k for k in range(int(np.floor(mk)) + 1, int(np.ceil(x))) if k > mk and k < x]
        pw = 0.5 + sum(mass_at(k) for k in ks); pp = mass_at(int(mk)) if float(mk).is_integer() else 0.0
        side = +1
    pl = max(0.0, 1 - pw - pp)
    return (pw - 1.1 * pl) / 1.1, side

def settle(side, mk, fm):
    """side +1 = origin favorite laying mk; -1 = dog taking mk. returns +1 win, 0 push, -1.1 loss (units)."""
    d = fm - mk
    if d == 0: return 0.0
    return 1.0 if (d > 0) == (side > 0) else -1.1

rows = []
for T in [0.5, 1.0, 1.5]:
    out = {}
    for rule in ["A", "B", "C"]:
        units, n, w, l, p = 0.0, 0, 0, 0, 0
        for x, mk, fm in zip(test.xq, test.mk, test.fm):
            r = pub(rule, x)
            if r - mk >= T: side = +1
            elif mk - r >= T: side = -1
            else: continue
            u = settle(side, mk, fm); units += u; n += 1; w += u > 0; l += u < 0; p += u == 0
        out[rule] = (n, w, l, p, units)
    # D: probability space, threshold in EV units chosen so bet count ~ matches A at this T
    for ev_min in [0.02, 0.04, 0.06]:
        units, n, w, l, p = 0.0, 0, 0, 0, 0
        for x, mk, fm in zip(test.x, test.mk, test.fm):
            ev, side = ev_prob_space(x, mk)
            if ev < ev_min: continue
            u = settle(side, mk, fm); units += u; n += 1; w += u > 0; l += u < 0; p += u == 0
        out[f"D(ev>={ev_min})"] = (n, w, l, p, units)
    for k, (n, w, l, p, units) in out.items():
        rows.append(dict(T=T, rule=k, bets=n, W=w, L=l, P=p, win_rate=w / (w + l) if w + l else np.nan, units=units, roi_per_bet=units / (1.1 * n) if n else np.nan))
print("\n=== OOS 2022-25 backtest of rules (bets vs market close; units at -110) ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# games where A and B (or A and C) make DIFFERENT decisions
print("\n=== Games where the rule changes the decision (OOS 2022-25) ===")
for T in [0.5, 1.0]:
    for alt in ["B", "C"]:
        diff_units_A, diff_units_alt, n = 0.0, 0.0, 0
        for x, mk, fm in zip(test.xq, test.mk, test.fm):
            dA = (+1 if pub("A", x) - mk >= T else (-1 if mk - pub("A", x) >= T else 0))
            dB = (+1 if pub(alt, x) - mk >= T else (-1 if mk - pub(alt, x) >= T else 0))
            if dA == dB: continue
            n += 1
            if dA: diff_units_A += settle(dA, mk, fm)
            if dB: diff_units_alt += settle(dB, mk, fm)
        print(f"T={T} A vs {alt}: decision differs in {n} games | units A={diff_units_A:+.1f} | units {alt}={diff_units_alt:+.1f}")
print("\nShare of OOS games whose quarter-rounded origin lands exactly on 2.75/3.25/6.75/7.25:", round(test.xq.isin([2.75, 3.25, 6.75, 7.25]).mean(), 3), "of", len(test))
