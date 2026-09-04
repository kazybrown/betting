"""CRITIC 04 (theory qb-3): Week-1 'new starter' (+2.24 vs market, n=115 events).  The expert's flag 'week-1 starter != previous
season's most-starts QB' also fires for a RETURNING franchise QB whose previous season was lost to injury (e.g. Brady 2009,
Rodgers 2018/2024, Burrow 2024 ...).  Split into returning-to-team (started for this team in a prior season) vs genuinely
new-to-team (never started for this team before); leave-one-season-out on the pooled estimate; qb_id-keyed 'first career
start'.  Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.mkt_spread.notna()].copy()
g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows = []
for side in ["home", "away"]:
    t = g_all[["gid", "season", "week", "game_type", "gdate", f"{side}_team", f"{side}_qb_name", f"{side}_qb_id"]].copy()
    t.columns = ["gid", "season", "week", "game_type", "gdate", "team", "qb", "qb_id"]; t["side"] = side; rows.append(t)
tg = pd.concat(rows).sort_values(["gdate", "gid"]).reset_index(drop=True); tg["team"] = tg.team.map(norm)
tg["career_id"] = tg.groupby("qb_id").cumcount()
# prior starts for THIS team in PRIOR seasons
tg = tg.sort_values(["team", "gdate"]).reset_index(drop=True)
tg["team_qb_starts_before"] = tg.groupby(["team", "qb"]).cumcount()
first_season_with_team = tg.groupby(["team", "qb"]).season.transform("min")
tg["returning"] = ((tg.team_qb_starts_before > 0) & (first_season_with_team < tg.season)).astype(int)
for side in ["home", "away"]:
    s = tg[tg.side == side].set_index("gid")
    m[f"{side}_returning"] = m.gid.map(s.returning).fillna(0).astype(int)
    m[f"{side}_career_id"] = m.gid.map(s.career_id)
    m[f"{side}_new"] = (m[f"{side}_qb"] != m[f"{side}_prev_season_primary"]).astype(int)
    m[f"{side}_new_ret"] = ((m[f"{side}_new"] == 1) & (m[f"{side}_returning"] == 1)).astype(int)
    m[f"{side}_new_fresh"] = ((m[f"{side}_new"] == 1) & (m[f"{side}_returning"] == 0)).astype(int)
    m[f"{side}_first_id"] = (m[f"{side}_career_id"] == 0).astype(int)
for f in ["new", "new_ret", "new_fresh", "first_id"]:
    m[f"net_{f}"] = m[f"home_{f}"] - m[f"away_{f}"]
w1 = m[(m.game_type == "REG") & (m.week == 1)].copy()
print("Week-1 games:", len(w1), " sign check corr(mkt_spread, margin):", round(np.corrcoef(w1.mkt_spread, w1.margin)[0, 1], 3))
print("team-events: new=%d  of which returning-to-team=%d, genuinely new-to-team=%d; first career start (id-keyed)=%d" % (
    int(w1.home_new.sum() + w1.away_new.sum()), int(w1.home_new_ret.sum() + w1.away_new_ret.sum()), int(w1.home_new_fresh.sum() + w1.away_new_fresh.sum()), int(w1.home_first_id.sum() + w1.away_first_id.sum())))
def coef(y, x, d):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())
def ats(d, x):
    d = d[(d[x] != 0)]; win = ((d[x] == 1) & (d.resid_mkt > 0)) | ((d[x] == -1) & (d.resid_mkt < 0)); push = d.resid_mkt == 0
    w = int(win.sum()); l = int((~win & ~push).sum()); return w, l, int(push.sum()), stats.binomtest(w, max(1, w + l), 0.5).pvalue if w + l else 1
print("\nresid vs MARKET close (+ = flagged team BEAT the number), week 1:")
for f, lab in [("new", "new starter (expert flag)"), ("new_ret", "  returning franchise QB (started for team in a prior season)"), ("new_fresh", "  genuinely new to team"), ("first_id", "first career start, id-keyed")]:
    for era, d in [("ALL", w1), ("<=2021", w1[w1.season <= 2021]), ("2022-25", w1[w1.season >= 2022])]:
        b, se, p, n = coef("resid_mkt", f"net_{f}", d); w, l, pu, pv = ats(d, f"net_{f}")
        print(f"  {lab:<62s} {era:<8s} coef={b:+.2f}±{se:.2f} p={p:.3f} n_event={n:3d}  BACK-flagged ATS {w}-{l}-{pu} (p={pv:.2f})")
    b, se, p, n = coef("mkt_minus_noqb", f"net_{f}", w1); print(f"  {'':<62s} market move vs no-QB line (penalty-signed) = {b:+.2f}±{se:.2f}")
print("\nleave-one-season-out of the pooled week-1 'new starter' coefficient:")
jk = [coef("resid_mkt", "net_new", w1[w1.season != yr])[0] for yr in sorted(w1.season.unique())]
print("  range [%.2f, %.2f]; seasons where the pooled p<0.05 survives dropping that season: %d/%d" % (min(jk), max(jk), sum(coef("resid_mkt", "net_new", w1[w1.season != yr])[2] < 0.05 for yr in sorted(w1.season.unique())), w1.season.nunique()))
print("  per-season event mean (flagged team resid vs market):")
ev = w1[w1.net_new != 0].assign(r=lambda d: d.resid_mkt * d.net_new)
print(ev.groupby("season").r.agg(["size", "mean"]).round(1).T.to_string())
