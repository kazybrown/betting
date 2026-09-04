"""THEORY 3 (part B): matchup reallocation using play-by-play profile features
(explosive-play rate, red-zone TD rate, pace, EPA/play, pass rate), offense and defense-allowed,
rolling last-10 team-games strictly before the game (min 5). Fit 2009-2019 (nflscrapR), test
2023-2025 (nflfastR) - 2020-2022 pbp is not available locally.

Null: given market S and T, none of these predict the identity residuals r_home / r_away.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import load, mean_ci, over_rate, boot_ci

g = load(min_season=2009)
tg = pd.read_csv(Path(__file__).resolve().parent / "_pbp_teamgame.csv")
tg["date"] = pd.to_datetime(tg.game_date)

# ---- attach the games.csv gid to each team-game
g_old = g[["gid", "old_game_id"]].copy(); g_old["old_game_id"] = g_old.old_game_id.astype("Int64")
tg["old_game_id"] = tg.old_game_id.astype("Int64")
a = tg[tg.old_game_id > 0].merge(g_old, on="old_game_id", how="inner")
b = tg[tg.old_game_id < 0].copy(); b["gid"] = b.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
b = b[b.gid.isin(g.gid)]
tg = pd.concat([a, b], ignore_index=True)
print(f"team-games joined to games.csv: {len(tg)} ({tg.season.min()}-{tg.season.max()})")

SUMS = ["plays", "expl20", "expl15", "epa_sum", "succ", "pass_plays", "pace_secs", "pace_plays", "rz_trips", "rz_td"]
W, MINN = 10, 5

def rolling_side(df, key):
    """rolling prior sums per team (offense: key='team'; defense allowed: key='opp').
    Windows are reset at the 2019 -> 2023 data gap (era) so 2023 W1-W5 do not borrow 2019 games."""
    d = df.sort_values([key, "date"]).copy()
    d["era"] = (d.season >= 2023).astype(int)
    out = d[[key, "gid"]].copy()
    for c in SUMS:
        out[c] = d.groupby([key, "era"])[c].transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).sum())
    out = out.rename(columns={key: "t"})
    out["expl"] = out.expl20 / out.plays
    out["expl15r"] = out.expl15 / out.plays
    out["epa"] = out.epa_sum / out.plays
    out["succr"] = out.succ / out.plays
    out["passr"] = out.pass_plays / out.plays
    out["pace"] = out.pace_secs / out.pace_plays          # seconds per play (higher = slower)
    out["rz"] = out.rz_td / out.rz_trips                  # red-zone TD rate
    out["ppg"] = out.plays / W                            # (approx) plays per game
    return out[["t", "gid", "expl", "expl15r", "epa", "succr", "passr", "pace", "rz", "ppg"]]

off = rolling_side(tg, "team")
de = rolling_side(tg, "opp")
F = ["expl", "expl15r", "epa", "succr", "passr", "pace", "rz", "ppg"]
h_off = off.rename(columns={"t": "home", **{f: f"ho_{f}" for f in F}})
a_off = off.rename(columns={"t": "away", **{f: f"ao_{f}" for f in F}})
h_def = de.rename(columns={"t": "home", **{f: f"hd_{f}" for f in F}})
a_def = de.rename(columns={"t": "away", **{f: f"ad_{f}" for f in F}})
m = g.merge(h_off, on=["gid", "home"]).merge(a_off, on=["gid", "away"]).merge(h_def, on=["gid", "home"]).merge(a_def, on=["gid", "away"])
m = m.dropna(subset=[c for c in m.columns if c[:3] in ("ho_", "ao_", "hd_", "ad_")]).copy()
m["sp_err"] = m.margin + m.S; m["tot_err"] = m.total_pts - m["T"]
m["train"] = m.season <= 2019; m["test"] = m.season >= 2023
tr, te = m[m.train], m[m.test]
print(f"games with full features: {len(m)} (train 2009-2019 n={len(tr)}, test 2023-2025 n={len(te)})")

# z-score with TRAIN moments
Z = {}
for c in [c for c in m.columns if c[:3] in ("ho_", "ao_", "hd_", "ad_")]:
    mu, sd = tr[c].mean(), tr[c].std()
    m[c + "_z"] = (m[c] - mu) / sd
tr, te = m[m.train], m[m.test]
print("train feature means: " + ", ".join(f"{f}: off {tr['ho_'+f].mean():.3f} / def {tr['hd_'+f].mean():.3f}" for f in ["expl", "rz", "pace", "epa"]))

# ---- matchup indices (z units): higher = more points expected for that offense
m["expl_home"] = m.ho_expl_z + m.ad_expl_z;  m["expl_away"] = m.ao_expl_z + m.hd_expl_z
m["rz_home"] = m.ho_rz_z + m.ad_rz_z;        m["rz_away"] = m.ao_rz_z + m.hd_rz_z
m["epa_home"] = m.ho_epa_z + m.ad_epa_z;     m["epa_away"] = m.ao_epa_z + m.hd_epa_z
m["pace_avg"] = -(m.ho_pace_z + m.ao_pace_z) / 2         # + = faster than avg (fewer sec/play)
m["pace_mismatch"] = (m.ho_pace_z - m.ao_pace_z).abs()
m["pace_home_rel"] = -(m.ho_pace_z - m.ao_pace_z)         # + = home faster than away
m["pass_home"] = m.ho_passr_z; m["pass_away"] = m.ao_passr_z
tr, te = m[m.train], m[m.test]

print("\n--- sanity: features carry information the market prices (margin on S + indices) ---")
r = smf.ols("margin ~ S + epa_home + epa_away + expl_home + expl_away", tr).fit(cov_type="HC1")
print("  " + ", ".join(f"{k} {r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in ["S", "epa_home", "epa_away", "expl_home", "expl_away"]))

print("\n--- identity residuals on matchup indices (TRAIN 2009-2019, HC1) ---")
specs = {
    "r_home": "r_home ~ expl_home + rz_home + epa_home + pace_avg + pace_mismatch + pass_home + S + T",
    "r_away": "r_away ~ expl_away + rz_away + epa_away + pace_avg + pace_mismatch + pass_away + S + T",
    "tot_err": "tot_err ~ I(expl_home+expl_away) + I(rz_home+rz_away) + I(epa_home+epa_away) + pace_avg + pace_mismatch + I(pass_home+pass_away) + S + T",
    "sp_err": "sp_err ~ I(expl_home-expl_away) + I(rz_home-rz_away) + I(epa_home-epa_away) + pace_home_rel + I(pass_home-pass_away) + S + T",
}
fits = {}
for y, f in specs.items():
    r = smf.ols(f, tr).fit(cov_type="HC1"); fits[y] = r
    print(f"TRAIN {y:>7s}: " + ", ".join(f"{k.replace('I(','').replace(')','')} {r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in r.params.index if k not in ("Intercept", "S", "T")) + f" | R2={r.rsquared:.4f}")
print("\n--- same regressions on TEST 2023-2025 (independent check, NOT a fit) ---")
for y, f in specs.items():
    r = smf.ols(f, te).fit(cov_type="HC1")
    print(f"TEST  {y:>7s}: " + ", ".join(f"{k.replace('I(','').replace(')','')} {r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in r.params.index if k not in ("Intercept", "S", "T")) + f" | R2={r.rsquared:.4f}")

# ---- single-factor tests with per-SD effect sizes and decile over/under rates (train + test)
print("\n--- single-factor: residual per 1 SD of index, and top/bottom-decile over rate vs identity ---")
for X, y, sc, tt in [("expl_home", "r_home", "home_score", "home_tt"), ("expl_away", "r_away", "away_score", "away_tt"),
                     ("rz_home", "r_home", "home_score", "home_tt"), ("rz_away", "r_away", "away_score", "away_tt"),
                     ("epa_home", "r_home", "home_score", "home_tt"), ("epa_away", "r_away", "away_score", "away_tt")]:
    for nm, d in [("TRAIN", tr), ("TEST", te)]:
        r = smf.ols(f"{y} ~ {X} + S + T", d).fit(cov_type="HC1")
        q = d[X].quantile([0.1, 0.9]); top = d[d[X] >= q[0.9]]; bot = d[d[X] <= q[0.1]]
        ot, _ = over_rate(top[sc], top[tt]); ob, _ = over_rate(bot[sc], bot[tt])
        print(f"{nm} {X:>9s} -> {y}: {r.params[X]*d[X].std():+.3f} pts/SD (p={r.pvalues[X]:.3f}, n={len(d)}) | top-decile resid {top[y].mean():+.2f} P(over) {ot:.3f} (n={len(top)}) | bottom-decile resid {bot[y].mean():+.2f} P(over) {ob:.3f}")

# pace -> total
for nm, d in [("TRAIN", tr), ("TEST", te)]:
    r = smf.ols("tot_err ~ pace_avg + pace_mismatch + S + T", d).fit(cov_type="HC1")
    print(f"{nm} tot_err ~ pace_avg + pace_mismatch: pace_avg {r.params['pace_avg']*d.pace_avg.std():+.3f} pts/SD (p={r.pvalues['pace_avg']:.3f}), pace_mismatch {r.params['pace_mismatch']*d.pace_mismatch.std():+.3f} pts/SD (p={r.pvalues['pace_mismatch']:.3f})")
    q = d.pace_avg.quantile([0.1, 0.9]); fast = d[d.pace_avg >= q[0.9]]; slow = d[d.pace_avg <= q[0.1]]
    of, _ = over_rate(fast.total_pts, fast["T"]); os_, _ = over_rate(slow.total_pts, slow["T"])
    print(f"      fastest-decile games: tot resid {fast.tot_err.mean():+.2f} P(over) {of:.3f} (n={len(fast)}) | slowest-decile: {slow.tot_err.mean():+.2f} P(over) {os_:.3f}")

# ---- OOS: reallocation rules fit on train applied to 2023-2025
print("\n--- OOS 2023-2025: team-total MAE, identity vs identity + reallocation fit on 2009-2019 ---")
rh = smf.ols("r_home ~ expl_home + rz_home + epa_home + pace_avg + pace_mismatch + pass_home", tr).fit()
ra = smf.ols("r_away ~ expl_away + rz_away + epa_away + pace_avg + pace_mismatch + pass_away", tr).fit()
ph = te.home_tt + rh.predict(te); pa = te.away_tt + ra.predict(te)
tot_d = 0.0
for side, sc, tt, p in [("home", "home_score", "home_tt", ph), ("away", "away_score", "away_tt", pa)]:
    e0 = (te[sc] - te[tt]).abs().values; e1 = (te[sc] - p).abs().values
    lo, hi = boot_ci(e0 - e1)
    print(f"  {side}: MAE identity {e0.mean():.3f} -> +full realloc {e1.mean():.3f}; dMAE {np.mean(e0-e1):+.3f} [{lo:+.3f},{hi:+.3f}] (realloc SD {np.std(p-te[tt]):.2f})")
# sum-preserving single index: delta = k*(explosive matchup differential)
for X in ["I(expl_home-expl_away)", "I(epa_home-epa_away)", "I(rz_home-rz_away)"]:
    rs = smf.ols(f"sp_err ~ {X}", tr).fit(cov_type="HC1")
    k = rs.params.iloc[1]
    xd = te.eval(X[2:-1])
    delta = 0.5 * k * xd
    e0h = (te.home_score - te.home_tt).abs().values; e1h = (te.home_score - (te.home_tt + delta)).abs().values
    e0a = (te.away_score - te.away_tt).abs().values; e1a = (te.away_score - (te.away_tt - delta)).abs().values
    lo, hi = boot_ci((e0h - e1h + e0a - e1a) / 2)
    print(f"  sum-preserving realloc on {X}: train coef {k:+.4f} (p={rs.pvalues.iloc[1]:.3f}); OOS dMAE {np.mean((e0h-e1h+e0a-e1a)/2):+.4f} [{lo:+.4f},{hi:+.4f}]; realloc SD {delta.std():.3f}")

# ---- what magnitude WOULD be justified if one took the train point estimates at face value?
print("\n--- implied reallocation magnitude at the 90th percentile of each index (train coefs, r_home spec) ---")
for X in ["expl_home", "rz_home", "epa_home"]:
    r = smf.ols(f"r_home ~ {X} + S + T", tr).fit(cov_type="HC1")
    p90 = tr[X].quantile(0.9)
    print(f"  {X}: coef {r.params[X]:+.3f} [{r.conf_int().loc[X,0]:+.3f},{r.conf_int().loc[X,1]:+.3f}] x p90 {p90:.2f} = {r.params[X]*p90:+.2f} pts")
