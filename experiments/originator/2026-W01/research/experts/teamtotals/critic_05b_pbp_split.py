"""CRITIC of TT3b (pbp matchup features). Rebuilds the expert's feature table (same code path),
then attacks: (a) split-half of the 2009-2019 train (2009-14 vs 2015-19): do the 'significant'
negative expl_home / epa_home effects hold in both halves? (b) rolling-origin reallocation inside
2013-2019 and 2024-2025 (fit on all prior seasons) - pooled OOS; (c) power: could the 2023-25 test
have detected the train effect? (d) placebo: the same regressions with features from the team's
NEXT 10 games (future information) - should show a large effect if the features carry signal
about scoring at all, sanity-checking the pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from common import load, boot_ci

g = load(min_season=2009, verbose=False)
tg = pd.read_csv(Path(__file__).resolve().parent / "_pbp_teamgame.csv"); tg["date"] = pd.to_datetime(tg.game_date)
g_old = g[["gid", "old_game_id"]].copy(); g_old["old_game_id"] = g_old.old_game_id.astype("Int64"); tg["old_game_id"] = tg.old_game_id.astype("Int64")
a = tg[tg.old_game_id > 0].merge(g_old, on="old_game_id", how="inner")
b = tg[tg.old_game_id < 0].copy(); b["gid"] = b.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_"); b = b[b.gid.isin(g.gid)]
tg = pd.concat([a, b], ignore_index=True)
SUMS = ["plays", "expl20", "epa_sum", "rz_trips", "rz_td", "pace_secs", "pace_plays"]
W, MINN = 10, 5

def rolling_side(df, key, future=False):
    d = df.sort_values([key, "date"]).copy(); d["era"] = (d.season >= 2023).astype(int)
    out = d[[key, "gid"]].copy()
    for c in SUMS:
        if future:
            out[c] = d.groupby([key, "era"])[c].transform(lambda s: s[::-1].shift(1).rolling(W, min_periods=MINN).sum()[::-1])
        else:
            out[c] = d.groupby([key, "era"])[c].transform(lambda s: s.shift(1).rolling(W, min_periods=MINN).sum())
    out = out.rename(columns={key: "t"})
    out["expl"] = out.expl20 / out.plays; out["epa"] = out.epa_sum / out.plays; out["rz"] = out.rz_td / out.rz_trips; out["pace"] = out.pace_secs / out.pace_plays
    return out[["t", "gid", "expl", "epa", "rz", "pace"]]

def assemble(future=False):
    off = rolling_side(tg, "team", future); de = rolling_side(tg, "opp", future); F = ["expl", "epa", "rz", "pace"]
    m = (g.merge(off.rename(columns={"t": "home", **{f: f"ho_{f}" for f in F}}), on=["gid", "home"])
          .merge(off.rename(columns={"t": "away", **{f: f"ao_{f}" for f in F}}), on=["gid", "away"])
          .merge(de.rename(columns={"t": "home", **{f: f"hd_{f}" for f in F}}), on=["gid", "home"])
          .merge(de.rename(columns={"t": "away", **{f: f"ad_{f}" for f in F}}), on=["gid", "away"]))
    m = m.dropna(subset=[c for c in m.columns if c[:3] in ("ho_", "ao_", "hd_", "ad_")]).copy()
    m["sp_err"] = m.margin + m.S; m["tot_err"] = m.total_pts - m["T"]
    tr = m[m.season <= 2019]
    for c in [c for c in m.columns if c[:3] in ("ho_", "ao_", "hd_", "ad_")]:
        m[c + "_z"] = (m[c] - tr[c].mean()) / tr[c].std()
    m["expl_home"] = m.ho_expl_z + m.ad_expl_z; m["expl_away"] = m.ao_expl_z + m.hd_expl_z
    m["epa_home"] = m.ho_epa_z + m.ad_epa_z; m["epa_away"] = m.ao_epa_z + m.hd_epa_z
    m["rz_home"] = m.ho_rz_z + m.ad_rz_z; m["rz_away"] = m.ao_rz_z + m.hd_rz_z
    return m

m = assemble()
print(f"games with features: {len(m)} (2009-2019: {(m.season<=2019).sum()}, 2023-25: {(m.season>=2023).sum()})")
print("\n(a) split-half of train: per-SD effect on identity residual (single-factor spec + S + T, HC1)")
halves = [("2009-14", m[(m.season >= 2009) & (m.season <= 2014)]), ("2015-19", m[(m.season >= 2015) & (m.season <= 2019)]), ("2023-25", m[m.season >= 2023])]
for X, y in [("expl_home", "r_home"), ("epa_home", "r_home"), ("rz_home", "r_home"), ("expl_away", "r_away"), ("epa_away", "r_away"), ("rz_away", "r_away")]:
    out = []
    for nm, d in halves:
        r = smf.ols(f"{y} ~ {X} + S + T", d).fit(cov_type="HC1"); sd = m[m.season <= 2019][X].std()
        out.append(f"{nm}: {r.params[X]*sd:+.3f}/SD (p={r.pvalues[X]:.2f}, n={len(d)})")
    print(f"  {X:>9s} -> {y}: " + " | ".join(out))

print("\n(b) rolling-origin reallocation (fit r_home/r_away ~ 3 indices on all prior seasons with features), pooled OOS 2013-2019 + 2024-2025")
rows = []
for s in list(range(2013, 2020)) + [2024, 2025]:
    a, b = m[m.season < s], m[m.season == s].copy()
    rh = smf.ols("r_home ~ expl_home + epa_home + rz_home", a).fit(); ra = smf.ols("r_away ~ expl_away + epa_away + rz_away", a).fit()
    b["d_h"] = np.abs(b.home_score - b.home_tt) - np.abs(b.home_score - (b.home_tt + rh.predict(b)))
    b["d_a"] = np.abs(b.away_score - b.away_tt) - np.abs(b.away_score - (b.away_tt + ra.predict(b)))
    b["rl_h"] = rh.predict(b); rows.append(b)
P = pd.concat(rows)
for col in ["d_h", "d_a"]:
    v = P[col].values; lo, hi = boot_ci(v); per = P.groupby("season")[col].mean()
    print(f"  {col}: pooled n={len(v)} dMAE (identity - rule; + = rule better) {v.mean():+.4f} [{lo:+.4f},{hi:+.4f}] | rule better in {(per>0).sum()}/{len(per)} seasons | realloc SD {P.rl_h.std():.2f}")
    print("     by season:", per.round(3).to_dict())

print("\n(c) power of the 2023-25 check for the train epa_home / expl_home effects")
tr, te = m[m.season <= 2019], m[m.season >= 2023]
for X in ["epa_home", "expl_home"]:
    r_tr = smf.ols(f"r_home ~ {X} + S + T", tr).fit(cov_type="HC1"); r_te = smf.ols(f"r_home ~ {X} + S + T", te).fit(cov_type="HC1")
    z = (r_te.params[X] - r_tr.params[X]) / np.sqrt(r_te.bse[X] ** 2 + r_tr.bse[X] ** 2)
    print(f"  {X}: train coef {r_tr.params[X]:+.3f} (se {r_tr.bse[X]:.3f}) vs test {r_te.params[X]:+.3f} (se {r_te.bse[X]:.3f}); z(test - train) = {z:+.2f} (p={2*stats.norm.sf(abs(z)):.3f}); power to detect train effect at 5% in test = {stats.norm.sf(1.96 - abs(r_tr.params[X])/r_te.bse[X]):.2f}")

print("\n(d) PLACEBO / pipeline sanity: features from the team's NEXT 10 games (future info) - must show strong effects if the pipeline works")
mf = assemble(future=True)
for X, y in [("epa_home", "r_home"), ("epa_away", "r_away"), ("expl_home", "r_home")]:
    r = smf.ols(f"{y} ~ {X} + S + T", mf[mf.season <= 2019]).fit(cov_type="HC1")
    print(f"  future {X:>9s} -> {y}: {r.params[X]*mf[X].std():+.3f} pts/SD (p={r.pvalues[X]:.4f}, n={int(r.nobs)})  [expected: large positive]")
