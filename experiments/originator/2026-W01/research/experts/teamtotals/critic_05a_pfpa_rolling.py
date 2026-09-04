"""CRITIC of TT3a (PF/PA matchup profiles). Attacks:
 (a) rolling-origin version of the reallocation (fit on all prior seasons, score season s, 2005-2025)
     -> pooled OOS n ~ 5,500 instead of 1,139, with season-block bootstrap
 (b) longer window (16) and season-to-date variant of the features
 (c) rating LEVELS instead of profiles: nfelo Elo (home/away pts vs avg) as regressors of the
     identity residuals (2009-2021 fit, 2022-25 check) - does the market split mis-price team quality?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from common import load, boot_ci, ROOT
sys.path.insert(0, str(ROOT))
from kit import load_nfelo

g = load(min_season=1999, verbose=False)
g["date"] = pd.to_datetime(g.gameday)
g["sp_err"] = g.margin + g.S; g["tot_err"] = g.total_pts - g["T"]

def build(W, MINN, std=False):
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        d = g[["gid", "season", "date", side, opp, f"{side}_score", f"{opp}_score"]].copy()
        d.columns = ["gid", "season", "date", "team", "opp", "pf", "pa"]; d["side"] = side; rows.append(d)
    L = pd.concat(rows).sort_values(["team", "date"]).reset_index(drop=True)
    if std:   # season-to-date: expanding mean within season, strictly prior, min MINN games
        L["off10"] = L.groupby(["team", "season"]).pf.transform(lambda s: s.shift(1).expanding(MINN).mean())
        L["def10"] = L.groupby(["team", "season"]).pa.transform(lambda s: s.shift(1).expanding(MINN).mean())
    else:
        L["off10"] = L.groupby("team").pf.transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).mean())
        L["def10"] = L.groupby("team").pa.transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).mean())
    lg = g.groupby("season").apply(lambda d: (d.home_score.sum() + d.away_score.sum()) / (2 * len(d)), include_groups=False)
    L["lg"] = L.season.map(lg.shift(1)).fillna(lg.iloc[0])
    L["off_p"] = L.off10 - L.lg; L["def_p"] = L.def10 - L.lg
    h = L[L.side == "home"][["gid", "off_p", "def_p"]].rename(columns={"off_p": "h_off", "def_p": "h_def"})
    a = L[L.side == "away"][["gid", "off_p", "def_p"]].rename(columns={"off_p": "a_off", "def_p": "a_def"})
    m = g.merge(h, on="gid").merge(a, on="gid").dropna(subset=["h_off", "h_def", "a_off", "a_def"]).copy()
    m["X_home"] = m.h_off + m.a_def; m["X_away"] = m.a_off + m.h_def
    return m

rng = np.random.default_rng(0)
def season_block_ci(vals, seasons, n=2000):
    u = np.unique(seasons); grp = {s: vals[seasons == s] for s in u}; bs = []
    for _ in range(n):
        pick = rng.choice(u, len(u), replace=True); bs.append(np.concatenate([grp[s] for s in pick]).mean())
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

for label, kw in [("last-10 (expert)", dict(W=10, MINN=5)), ("last-16", dict(W=16, MINN=8)), ("season-to-date (min 4)", dict(W=0, MINN=4, std=True))]:
    m = build(**kw)
    rows = []
    for s in range(2005, 2026):
        a, b = m[m.season < s], m[m.season == s].copy()
        rh = smf.ols("r_home ~ h_off + h_def + a_off + a_def", a).fit(); ra = smf.ols("r_away ~ h_off + h_def + a_off + a_def", a).fit()
        b["d_h"] = np.abs(b.home_score - b.home_tt) - np.abs(b.home_score - (b.home_tt + rh.predict(b)))
        b["d_a"] = np.abs(b.away_score - b.away_tt) - np.abs(b.away_score - (b.away_tt + ra.predict(b)))
        rs = smf.ols("sp_err ~ I(X_home - X_away)", a).fit(); k = rs.params.iloc[1]
        delta = 0.5 * k * (b.X_home - b.X_away)
        b["d_sp"] = 0.5 * ((np.abs(b.home_score - b.home_tt) - np.abs(b.home_score - (b.home_tt + delta))) + (np.abs(b.away_score - b.away_tt) - np.abs(b.away_score - (b.away_tt - delta))))
        b["realloc_h"] = rh.predict(b); b["delta_sp"] = delta
        rows.append(b)
    P = pd.concat(rows)
    print(f"\n[{label}] rolling-origin 2005-2025 pooled n={len(P)} (feature rows {len(m)})")
    for col, nm in [("d_h", "home 4-profile realloc"), ("d_a", "away 4-profile realloc"), ("d_sp", "sum-preserving single index")]:
        v = P[col].values; lo, hi = boot_ci(v); slo, shi = season_block_ci(v, P.season.values)
        per = P.groupby("season")[col].mean()
        print(f"   {nm:>28s}: dMAE (identity - rule; + = rule better) {v.mean():+.4f} game-CI [{lo:+.4f},{hi:+.4f}] season-block CI [{slo:+.4f},{shi:+.4f}] | rule better in {(per>0).sum()}/{len(per)} seasons | realloc SD {P['realloc_h' if col=='d_h' else 'delta_sp'].std():.2f}")
    # in the expert's split for reference
    tr, te = m[m.season <= 2021], m[m.season >= 2022]
    for y in ["r_home", "r_away", "sp_err", "tot_err"]:
        r = smf.ols(f"{y} ~ h_off + h_def + a_off + a_def + S + T", tr).fit(cov_type="HC1")
        print(f"   TRAIN {y:>7s}: " + ", ".join(f"{k} {r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in ["h_off", "h_def", "a_off", "a_def"]) + f" | R2={r.rsquared:.4f}")

print("\n(c) rating LEVELS: identity residuals ~ nfelo home/away rating (pts vs avg) + S + T, 2009-2021 fit / 2022-25 check")
n = load_nfelo()[["gid", "home_pts_vs_avg", "away_pts_vs_avg", "hfa_pts"]]
mm = g.merge(n, on="gid", how="inner").dropna(subset=["home_pts_vs_avg", "away_pts_vs_avg"])
print(f"   merged n={len(mm)} seasons {mm.season.min()}-{mm.season.max()}")
for nm, d in [("TRAIN 2009-21", mm[mm.season <= 2021]), ("TEST 2022-25", mm[mm.season >= 2022])]:
    for y in ["r_home", "r_away", "sp_err", "tot_err"]:
        r = smf.ols(f"{y} ~ home_pts_vs_avg + away_pts_vs_avg + S + T", d).fit(cov_type="HC1")
        print(f"   {nm} {y:>7s} (n={len(d)}): home_elo {r.params['home_pts_vs_avg']:+.3f} (p={r.pvalues['home_pts_vs_avg']:.2f}), away_elo {r.params['away_pts_vs_avg']:+.3f} (p={r.pvalues['away_pts_vs_avg']:.2f}) | R2={r.rsquared:.4f}")
