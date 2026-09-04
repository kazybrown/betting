"""01: LEAK-FREE per-game pace / style features + validation.
For every team-game feature in _teamgame.csv, three prior-only versions are built:
  {f}_r8   mean over the team's PRIOR 8 games (min 4), never crossing the 2019->2023 pbp gap;
           spp_* are weighted by their gap counts
  {f}_prev prior-season REG mean (NaN in 2009 and 2023 -- no prior pbp season)
  {f}_bl   blend = (K*prior + gp*ytd)/(K+gp), K=4, ytd = season-to-date mean BEFORE the game,
           prior = team prev-season mean, else league prev-season mean, else last league mean
Game table (_game_features.csv): one row per scored game 2009-2019 & 2023-2025 with h_/a_ features,
market lines, Elo (nfelo), and the totals expert's leak-free scoring proxies (pf/pa/qb/lg_blend/env)
imported from experts/totals/common.py so every expert measures increments against the same baseline.
Validation: my per-game plays / drives / pass rate / PROE vs Kevin Cole's team-game table
(2023-2025 weeks 1-4) and vs nflfastR pass_oe. Sign sanity checks per README.
"""
import sys
import warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experts" / "totals"))
from kit import load_games, norm
from common import build

K = 4
FEATS = ["plays", "def_plays", "drives", "ppd", "spp_neut", "spp_neut_run", "nh_neut", "pr_neut", "pr_all", "pr_wp",
         "proe", "proe_neut", "expl_off", "expl20_off", "expl_def", "expl20_def", "epa_off", "epa_def",
         "succ_off", "succ_def", "pts", "pts_allowed"]
WEIGHT = {"spp_neut": "spp_neut_n", "spp_neut_run": "spp_neut_run_n"}

tg = pd.read_csv(HERE / "_teamgame.csv")
tg["era"] = np.where(tg.season <= 2019, 1, 2)
tg["game_plays"] = tg.plays + tg.def_plays
FEATS.append("game_plays")
tg = tg.sort_values(["team", "season", "gameday", "gid"]).reset_index(drop=True)
grp_era = tg.groupby(["team", "era"])
grp_ss = tg.groupby(["team", "season"])
tg["gp"] = grp_ss.cumcount()
for f in FEATS:
    if f in WEIGHT:
        w = tg[WEIGHT[f]].fillna(0); num = (tg[f].fillna(0) * w)
        rn = grp_era[num.name if False else f].transform(lambda s: s)  # placeholder to keep index
        tg["_num"] = num; tg["_w"] = w
        rs = tg.groupby(["team", "era"])["_num"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).sum())
        rw = tg.groupby(["team", "era"])["_w"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).sum())
        tg[f + "_r8"] = rs / rw.replace(0, np.nan)
        ys = tg.groupby(["team", "season"])["_num"].transform(lambda s: s.shift(1).expanding().sum())
        yw = tg.groupby(["team", "season"])["_w"].transform(lambda s: s.shift(1).expanding().sum())
        tg[f + "_ytd"] = ys / yw.replace(0, np.nan)
    else:
        tg[f + "_r8"] = grp_era[f].transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
        tg[f + "_ytd"] = grp_ss[f].transform(lambda s: s.shift(1).expanding().mean())
tg = tg.drop(columns=["_num", "_w"], errors="ignore")
# prior-season means (REG) and league means
prev = tg[tg.game_type == "REG"].groupby(["team", "season"])[FEATS].mean().reset_index(); prev["season"] += 1
prev = prev.rename(columns={f: f + "_prev" for f in FEATS})
lg = tg[tg.game_type == "REG"].groupby("season")[FEATS].mean()
lg_prev = lg.copy(); lg_prev.index = lg_prev.index + 1
lg_prev = lg_prev.rename(columns={f: f + "_lgprev" for f in FEATS}).reset_index()
tg = tg.merge(prev, on=["team", "season"], how="left").merge(lg_prev, on="season", how="left")
last_lg = lg.loc[2019]   # fallback prior for 2023 (no 2022 pbp): the last available league season
for f in FEATS:
    prior = tg[f + "_prev"].fillna(tg[f + "_lgprev"]).fillna(last_lg[f])
    ytd = tg[f + "_ytd"].fillna(prior)
    tg[f + "_bl"] = (K * prior + tg.gp * ytd) / (K + tg.gp)
    # league-relative versions (season-to-date league mean is not leak-free enough for r8; use lg prev / last)
    lgp = tg[f + "_lgprev"].fillna(last_lg[f])
    tg[f + "_r8d"] = tg[f + "_r8"] - lgp
    tg[f + "_bld"] = tg[f + "_bl"] - lgp
tg = tg.sort_values(["season", "gameday", "gid", "is_home"]).reset_index(drop=True)
tg.to_csv(HERE / "_teamgame_feats.csv", index=False)

# ---------------- game table ----------------
m = build(K_team=6, K_lg=128, verbose=False)   # totals expert's leak-free baseline table (pf/pa/qb/lg_blend/env)
m = m[m.season.isin(list(range(2009, 2020)) + [2023, 2024, 2025])].copy()
suf = ["_r8", "_r8d", "_bl", "_bld", "_prev"]
cols = ["gid", "team"] + [f + s for f in FEATS for s in suf] + ["gp"]
sub = tg[cols]
m = m.merge(sub.rename(columns={c: "h_" + c for c in cols if c not in ("gid", "team")}).rename(columns={"team": "home"}), on=["gid", "home"], how="left")
m = m.merge(sub.rename(columns={c: "a_" + c for c in cols if c not in ("gid", "team")}).rename(columns={"team": "away"}), on=["gid", "away"], how="left")
# realized game-level quantities from pbp (for descriptive checks only)
act = tg[["gid", "team", "plays", "drives", "spp_neut", "pr_neut", "proe", "expl_off", "ppd"]]
m = m.merge(act.rename(columns={c: "h_" + c + "_act" for c in act.columns if c not in ("gid", "team")}).rename(columns={"team": "home"}), on=["gid", "home"], how="left")
m = m.merge(act.rename(columns={c: "a_" + c + "_act" for c in act.columns if c not in ("gid", "team")}).rename(columns={"team": "away"}), on=["gid", "away"], how="left")
m["implied_home_tt"] = m.mkt_total / 2 - m.mkt_spread / 2
m["implied_away_tt"] = m.mkt_total / 2 + m.mkt_spread / 2
m["train"] = m.season <= 2019; m["test"] = m.season >= 2023
m.to_csv(HERE / "_game_features.csv", index=False)
print(f"game table: {len(m)} games | REG with market total & r8 features both sides: "
      f"{((m.game_type=='REG') & m.mkt_total.notna() & m.h_plays_r8.notna() & m.a_plays_r8.notna()).sum()} "
      f"(train {((m.train)&(m.game_type=='REG')&m.mkt_total.notna()&m.h_plays_r8.notna()&m.a_plays_r8.notna()).sum()}, "
      f"test {((m.test)&(m.game_type=='REG')&m.mkt_total.notna()&m.h_plays_r8.notna()&m.a_plays_r8.notna()).sum()})")
print("r8 coverage by season:", m.groupby("season").apply(lambda x: (x.h_plays_r8.notna() & x.a_plays_r8.notna()).mean()).round(2).to_dict())

# ---------------- sanity ----------------
r = m[(m.game_type == "REG") & m.mkt_total.notna()]
print(f"\nSIGN CHECKS: corr(mkt_spread, margin) = {np.corrcoef(r.mkt_spread, r.margin)[0,1]:+.3f} (must be strongly NEGATIVE) | "
      f"corr(mkt_total, total_pts) = {np.corrcoef(r.mkt_total, r.total_pts)[0,1]:+.3f} (positive)")
h = r[["implied_home_tt", "home_score", "implied_away_tt", "away_score"]].dropna()
print(f"implied team totals: corr(home implied, home pts) = {np.corrcoef(h.implied_home_tt, h.home_score)[0,1]:+.3f}; "
      f"corr(away implied, away pts) = {np.corrcoef(h.implied_away_tt, h.away_score)[0,1]:+.3f} (both positive)")
a = tg.dropna(subset=["spp_neut", "plays"])
print(f"team-game: corr(spp_neut, plays) = {np.corrcoef(a.spp_neut, a.plays)[0,1]:+.3f} (negative: slower -> fewer plays) | "
      f"corr(pr_all, plays) = {np.corrcoef(a.pr_all, a.plays)[0,1]:+.3f} | corr(proe, pr_all) = {np.corrcoef(a.proe, a.pr_all)[0,1]:+.3f}")
rr = r.dropna(subset=["h_plays_r8", "a_plays_r8"])
gp_act = rr.h_plays_act + rr.a_plays_act
print(f"prior pace -> realized game plays (REG, both sides r8): corr(h_plays_r8+a_plays_r8, actual game plays) = "
      f"{np.corrcoef(rr.h_plays_r8 + rr.a_plays_r8, gp_act)[0,1]:+.3f} | corr(h_game_plays_r8+a_game_plays_r8, actual) = "
      f"{np.corrcoef(rr.h_game_plays_r8 + rr.a_game_plays_r8, gp_act)[0,1]:+.3f} | corr(spp avg r8, actual plays) = "
      f"{np.corrcoef((rr.h_spp_neut_r8 + rr.a_spp_neut_r8)/2, gp_act)[0,1]:+.3f} | corr(actual game plays, total_pts) = {np.corrcoef(gp_act, rr.total_pts)[0,1]:+.3f}")
print("year-to-year stability of team-season means (corr season t vs t+1, REG):")
ts = tg[tg.game_type == "REG"].groupby(["team", "season"])[["plays", "spp_neut", "spp_neut_run", "pr_neut", "proe", "expl_off", "expl_def", "ppd", "epa_off"]].mean().reset_index()
ts2 = ts.copy(); ts2["season"] -= 1
j = ts.merge(ts2, on=["team", "season"], suffixes=("", "_next"))
print("  " + "  ".join(f"{c}={np.corrcoef(j[c], j[c+'_next'])[0,1]:+.2f}" for c in ["plays", "spp_neut", "spp_neut_run", "pr_neut", "proe", "expl_off", "expl_def", "ppd", "epa_off"]))
print("split-half reliability within season (odd vs even games, team-season, REG):")
tg = tg.assign(_odd=tg.gp % 2)
sh = tg[tg.game_type == "REG"].groupby(["team", "season", "_odd"])[["plays", "spp_neut", "spp_neut_run", "pr_neut", "proe", "expl_off", "expl_def", "ppd", "epa_off"]].mean().unstack("_odd")
print("  " + "  ".join(f"{c}={np.corrcoef(sh[(c,0)], sh[(c,1)])[0,1]:+.2f}" for c in ["plays", "spp_neut", "spp_neut_run", "pr_neut", "proe", "expl_off", "expl_def", "ppd", "epa_off"]))

# ---------------- validation vs Kevin Cole team-game table ----------------
c = pd.read_csv(ROOT / "data" / "cole_team_games_2012_2025.csv")
c["team"] = c.team.map(norm)
c["gid"] = c.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
mine_cols = ["plays", "drives", "pr_all", "proe", "proe_neut", "pts_off", "ppd", "pts"]
v = c.merge(tg[["gid", "team"] + mine_cols].rename(columns={k: "my_" + k for k in mine_cols}), on=["gid", "team"], how="inner")
gd = tg.groupby("gid").drives.sum().rename("drives_game")
v = v.merge(gd, on="gid", how="left")
print(f"\nCOLE VALIDATION: matched team-games = {len(v)} (seasons {sorted(v.season.unique())})")
for mine, his, lab in [("my_plays", "plays", "plays (Cole counts all plays incl. penalties/ST?)"), ("drives_game", "game_drives", "game drives (both teams)"),
                       ("my_pr_all", "pass_rate", "pass rate (Cole in %)"), ("my_proe", "pass_over_exp", "PROE (all plays)"), ("my_proe_neut", "pass_over_exp", "PROE neutral"), ("my_pts", "score", "score")]:
    x = v[[mine, his]].dropna()
    print(f"  {lab:48s} corr={np.corrcoef(x[mine], x[his])[0,1]:+.3f}  mine mean={x[mine].mean():.2f} Cole mean={x[his].mean():.2f}")
f = tg[tg.season >= 2023].dropna(subset=["proe_fastr"])
print(f"nflfastR pass_oe vs my proe (2023-25 team-games n={len(f)}): corr={np.corrcoef(f.proe, f.proe_fastr)[0,1]:+.3f}; "
      f"team-season means corr={f.groupby(['team','season'])[['proe','proe_fastr']].mean().corr().iloc[0,1]:+.3f}")
