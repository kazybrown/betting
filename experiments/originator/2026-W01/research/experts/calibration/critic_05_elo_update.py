"""CRITIC 05 (T2/T4 leakage): are nfelo's Elo UPDATES market-informed? Controlled specifications.
critic_02 D found d_elo ~ e_own + (mkt_line - own_line) gives a coefficient of about -3 Elo per point of line gap.
That could be an artifact of Elo's non-linear update (win/loss * margin-of-victory multiplier that depends on the
pre-game Elo gap). Here:
 1. add the 538-style update term K*mult*(W - p_own) built from nfelo_dif_base and the result, plus own_line and interactions;
 2. non-parametric: within narrow bins of the own-forecast error e_own, does d_elo still move with the market gap?
 3. by era.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import ROOT
import sys; sys.path.insert(0, str(ROOT))
from kit import load_nfelo, load_games
pd.set_option("display.width", 220)
n = load_nfelo(); n = n[n.season <= 2025]
g = load_games()[["gid", "margin", "mkt_spread", "game_type"]]
x = n.merge(g, on="gid", how="inner"); x["nfelo_lin"] = -x.nfelo_dif_base / 25.0
long = []
for side, sgn in (("home", 1), ("away", -1)):
    long.append(pd.DataFrame({"team": x[side], "season": x.season, "week": x.week, "elo": x[f"starting_nfelo_{side}"], "dif_own": sgn * x.nfelo_dif_base,
                              "mg": sgn * x.margin, "own_line": sgn * x.nfelo_lin, "mkt_line": sgn * x.mkt_spread, "post": x.game_type != "REG"}))
L = pd.concat(long).sort_values(["team", "season", "week"])
L["elo_next"] = L.groupby(["team", "season"]).elo.shift(-1); L["d_elo"] = L.elo_next - L.elo
L = L[L.d_elo.notna() & L.mkt_line.notna() & ~L.post].copy()
L["e_own"] = L.mg + L.own_line; L["e_mkt"] = L.mg + L.mkt_line; L["diff_line"] = L.mkt_line - L.own_line
L["p_own"] = 1 / (1 + 10 ** (-L.dif_own / 400)); L["W"] = (L.mg > 0).astype(float) + 0.5 * (L.mg == 0)
dif_winner = np.where(L.mg >= 0, L.dif_own, -L.dif_own)
L["mult"] = np.log(np.abs(L.mg) + 1) * 2.2 / (0.001 * dif_winner + 2.2)
L["upd"] = L.mult * (L.W - L.p_own)
print(f"n={len(L)} team-games (REG, within-season next game). sd(d_elo)={L.d_elo.std():.1f}; mean |diff_line| = {L.diff_line.abs().mean():.2f} pts")
def fit(cols, label):
    r = sm.OLS(L.d_elo.values, sm.add_constant(L[cols].values.astype(float))).fit(cov_type="HC1")
    print(f"  {label:58s} R2={r.rsquared:.3f} | " + " ".join(f"{c}={r.params[i+1]:+.3f}±{1.96*r.bse[i+1]:.3f}" for i, c in enumerate(cols)))
    return r
print("\n== 1. Controlled regressions of the Elo change ==")
fit(["upd"], "d_elo ~ 538-style update K*mult*(W-p)")
fit(["upd", "diff_line"], "d_elo ~ upd + (mkt_line - own_line)")
L["e_own_x_line"] = L.e_own * L.own_line; L["e_own2"] = L.e_own * np.abs(L.e_own); L["winner"] = (L.mg > 0).astype(float)
fit(["e_own", "own_line", "e_own_x_line", "e_own2", "winner", "diff_line"], "d_elo ~ e_own + own_line + e_own*own_line + e_own|e_own| + win + diff")
fit(["upd", "e_own", "own_line", "e_own_x_line", "e_own2", "winner", "diff_line"], "  ... + upd")
r = fit(["upd", "e_own", "own_line", "e_own_x_line", "e_own2", "winner", "diff_line", "e_mkt"], "  ... + e_mkt (market error) as well")
print("\n== 2. Non-parametric: within narrow bins of the own-forecast error, slope of d_elo on the market gap (mkt_line - own_line) ==")
for lo, hi in ((-1, 1), (-3, 3), (3, 7), (-7, -3), (7, 14), (-14, -7)):
    d = L[(L.e_own >= lo) & (L.e_own < hi)]
    r = sm.OLS(d.d_elo.values, sm.add_constant(np.column_stack([d.e_own.values, d.diff_line.values]))).fit(cov_type="HC1")
    t = d.assign(tb=pd.qcut(d.diff_line, 3, labels=["mkt lower on team", "mid", "mkt higher on team"])).groupby("tb", observed=True).d_elo.mean().round(1).to_dict()
    print(f"  e_own in [{lo},{hi}): n={len(d)} slope on diff_line {r.params[2]:+.3f} ± {1.96*r.bse[2]:.3f} | mean d_elo by market-gap tercile: {t}")
print("\n== 3. By era, full-control model ==")
for era, (a, b) in {"2009-13": (2009, 2013), "2014-18": (2014, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025)}.items():
    d = L[(L.season >= a) & (L.season <= b)]
    cols = ["upd", "e_own", "own_line", "e_own_x_line", "e_own2", "winner", "diff_line"]
    r = sm.OLS(d.d_elo.values, sm.add_constant(d[cols].values.astype(float))).fit(cov_type="HC1")
    print(f"  {era}: diff_line coef {r.params[-1]:+.3f} ± {1.96*r.bse[-1]:.3f}  (R2 {r.rsquared:.3f}, n={int(r.nobs)})")
print("\nInterpretation: diff_line = market line minus nfelo's own line from the team's perspective (positive = the market likes the team LESS than nfelo does)."
      " A negative coefficient that survives the update controls means each game's rating update pulls the team's Elo toward the market's implied rating,"
      " i.e. nfelo's starting Elo is not market-blind.")
# implied fraction of the rating gap closed per game: 1 pt of line gap ~ 25 Elo of rating-dif gap ~ 12.5 Elo per team
print(f"per-game pull toward the market per point of line gap: {r.params[7]:.2f} Elo (from the last model) = {abs(r.params[7])/12.5:.0%} of the per-team rating gap implied by that point")
