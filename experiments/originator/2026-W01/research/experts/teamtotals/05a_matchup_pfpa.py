"""THEORY 3 (part A): matchup reallocation using rolling points-for / points-against profiles
from games.csv (1999-2025). Features are strictly prior to the game (last 10 team-games,
crossing seasons, min 5). Question: given the market S and T, is the identity's home/away
split predictable from offense/defense profiles?  Under an efficient market every
coefficient below should be ~0. Fit <= 2021, test 2022-2025.

Features (points vs rolling league average, so positive = more points):
  h_off, h_def (points ALLOWED, positive = weak defense), a_off, a_def
  X_home = h_off + a_def  (explosive offense vs weak defense, home scoring)
  X_away = a_off + h_def
  tempo_h = h_off + h_def (high-scoring games profile), tempo_a likewise
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import load, mean_ci, over_rate, boot_ci

g = load(min_season=1999)
g["date"] = pd.to_datetime(g.gameday)

# long team-game table for rolling
rows = []
for side, opp in [("home", "away"), ("away", "home")]:
    d = g[["gid", "season", "date", side, opp, f"{side}_score", f"{opp}_score"]].copy()
    d.columns = ["gid", "season", "date", "team", "opp", "pf", "pa"]
    d["side"] = side
    rows.append(d)
L = pd.concat(rows).sort_values(["team", "date"]).reset_index(drop=True)
W, MINN = 10, 5
L["off10"] = L.groupby("team").pf.transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).mean())
L["def10"] = L.groupby("team").pa.transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).mean())
# league average of team points, rolling by season-to-date (prior seasons for early weeks): use prior-season mean of team score
lg = g.groupby("season").apply(lambda d: (d.home_score.sum() + d.away_score.sum()) / (2 * len(d)), include_groups=False)
L["lg"] = L.season.map(lg.shift(1)).fillna(lg.iloc[0])
L["off_p"] = L.off10 - L.lg
L["def_p"] = L.def10 - L.lg

h = L[L.side == "home"][["gid", "off_p", "def_p"]].rename(columns={"off_p": "h_off", "def_p": "h_def"})
a = L[L.side == "away"][["gid", "off_p", "def_p"]].rename(columns={"off_p": "a_off", "def_p": "a_def"})
m = g.merge(h, on="gid").merge(a, on="gid")
m = m.dropna(subset=["h_off", "h_def", "a_off", "a_def"]).copy()
m["X_home"] = m.h_off + m.a_def
m["X_away"] = m.a_off + m.h_def
m["tempo_h"] = m.h_off + m.h_def
m["tempo_a"] = m.a_off + m.a_def
m["sp_err"] = m.margin + m.S
m["tot_err"] = m.total_pts - m["T"]
tr, te = m[m.train], m[m.test]
print(f"\nrows with features: {len(m)} (train {len(tr)}, test {len(te)}); feature SDs: h_off {m.h_off.std():.2f} h_def {m.h_def.std():.2f} X_home {m.X_home.std():.2f}")

# sanity: do the features carry rating information the market already prices?  (they should predict margin, and the market should absorb it)
r0 = smf.ols("margin ~ h_off + h_def + a_off + a_def", tr).fit()
r1 = smf.ols("margin ~ S + h_off + h_def + a_off + a_def", tr).fit(cov_type="HC1")
print(f"sanity: R2 margin~profiles = {r0.rsquared:.3f}; with S added, profile coefs: " + ", ".join(f"{k} {r1.params[k]:+.3f} (p={r1.pvalues[k]:.2f})" for k in ["h_off", "h_def", "a_off", "a_def"]))

# main: identity residuals on profiles
for y in ["r_home", "r_away", "tot_err", "sp_err"]:
    r = smf.ols(f"{y} ~ h_off + h_def + a_off + a_def + S + T", tr).fit(cov_type="HC1")
    print(f"TRAIN {y:>7s} ~ profiles + S + T : " + ", ".join(f"{k} {r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in ["h_off", "h_def", "a_off", "a_def"]) + f" | R2={r.rsquared:.4f}")
for y in ["r_home", "r_away"]:
    X = "X_home" if y == "r_home" else "X_away"
    r = smf.ols(f"{y} ~ {X} + S + T", tr).fit(cov_type="HC1")
    print(f"TRAIN {y:>7s} ~ {X} + S + T : {X} {r.params[X]:+.4f} (se {r.bse[X]:.4f}, p={r.pvalues[X]:.3f}) -> per-SD effect {r.params[X]*m[X].std():+.3f} pts")

# decile test: top vs bottom decile of X_home (explosive off vs weak def) -> residual and P(over) of identity home tt
for nm, d in [("TRAIN", tr), ("TEST", te)]:
    for X, y, sc, tt in [("X_home", "r_home", "home_score", "home_tt"), ("X_away", "r_away", "away_score", "away_tt")]:
        q = d[X].quantile([0.1, 0.9])
        top = d[d[X] >= q[0.9]]; bot = d[d[X] <= q[0.1]]
        mt = mean_ci(top[y]); mb = mean_ci(bot[y])
        ot, _ = over_rate(top[sc], top[tt]); ob, _ = over_rate(bot[sc], bot[tt])
        print(f"{nm} {X}: top decile (n={len(top)}) resid {mt[0]:+.2f} [{mt[1]:+.2f},{mt[2]:+.2f}] P(over)={ot:.3f} | bottom decile (n={len(bot)}) resid {mb[0]:+.2f} [{mb[1]:+.2f},{mb[2]:+.2f}] P(over)={ob:.3f}")

# OOS: reallocation rule fit on train (coefficients from the 4-profile regression), applied to 2022-2025
rh = smf.ols("r_home ~ h_off + h_def + a_off + a_def", tr).fit()
ra = smf.ols("r_away ~ h_off + h_def + a_off + a_def", tr).fit()
ph = te.home_tt + rh.predict(te); pa = te.away_tt + ra.predict(te)
for side, sc, tt, p in [("home", "home_score", "home_tt", ph), ("away", "away_score", "away_tt", pa)]:
    e0 = (te[sc] - te[tt]).abs().values; e1 = (te[sc] - p).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"OOS {side}: MAE identity {e0.mean():.3f} vs identity+profile-realloc {e1.mean():.3f}; dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}]; realloc SD {np.std(p-te[tt]):.2f}, |realloc|>0.5 in {np.mean(np.abs(p-te[tt])>0.5):.2f} of games")
# also the single-index version with sum preserved (pure reallocation = spread shift): delta = k*(X_home - X_away)
rs = smf.ols("sp_err ~ I(X_home - X_away)", tr).fit(cov_type="HC1")
k = rs.params.iloc[1]
print(f"\nsum-preserving reallocation: sp_err ~ (X_home - X_away): coef {k:+.4f} (p={rs.pvalues.iloc[1]:.3f}); train n={len(tr)}")
delta = 0.5 * k * (te.X_home - te.X_away)
e0h = (te.home_score - te.home_tt).abs().values; e1h = (te.home_score - (te.home_tt + delta)).abs().values
e0a = (te.away_score - te.away_tt).abs().values; e1a = (te.away_score - (te.away_tt - delta)).abs().values
lo, hi = boot_ci((e0h - e1h + e0a - e1a) / 2)
print(f"OOS sum-preserving realloc: mean dMAE (home+away)/2 = {np.mean((e0h-e1h+e0a-e1a)/2):+.4f} [{lo:+.4f},{hi:+.4f}]; realloc SD {delta.std():.3f}")
rt = smf.ols("tot_err ~ I(X_home + X_away)", tr).fit(cov_type="HC1")
print(f"total-side: tot_err ~ (X_home + X_away): coef {rt.params.iloc[1]:+.4f} (p={rt.pvalues.iloc[1]:.3f})")
rt2 = smf.ols("tot_err ~ I(X_home + X_away)", te).fit(cov_type="HC1")
print(f"   TEST same regression: coef {rt2.params.iloc[1]:+.4f} (p={rt2.pvalues.iloc[1]:.3f}) n={len(te)}")
