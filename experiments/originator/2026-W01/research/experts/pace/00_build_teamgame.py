"""00: Per TEAM-GAME pace / play-style / explosiveness features from play-by-play.
Sources: nflscrapR reg-season pbp 2009-2019 (join games.old_game_id) and nflfastR 2023-2025
(REG+POST, join games.game_id). 2020-2022 pbp is NOT available locally (README).

Definitions (all from pass/run scrimmage plays unless noted; kneels/spikes/penalty no-plays excluded):
  plays        offensive pass+run plays in the game
  drives       offensive drives (distinct drive ids with a non-kickoff/XP play by the offense)
  pts_off      points scored on offensive drives = sum over drives of
               (max posteam_score_post - min posteam_score)  -> ppd = pts_off / drives
  NEUTRAL situation = quarters 1-3, |score diff| <= 8, excluding the last 2 min of the 1st half
  spp_neut     game-clock seconds elapsed between consecutive offensive snaps of the same drive
               in neutral situations (0 < dt <= 60; timeouts/2-min/quarter breaks excluded)
  spp_neut_run same, but only gaps that follow a RUN play (clock kept running -> tempo, not style)
  nh_neut      no-huddle share of neutral pass/run plays
  pr_neut      pass share of neutral pass/run plays;  pr_all = pass share of all pass/run plays
  pr_wp        pass share with wp in [0.2,0.8] and quarters 1-3 (alternative 'neutral')
  proe         pass rate over expected (x100) from an expected-pass logit FIT ON 2009-2019 ONLY
               (features: down, distance, field position, score diff x time, half/game clock, wp)
  proe_neut    same, neutral situations only
  proe_fastr   nflfastR's own pass_oe mean (2023-2025 only; validation of proe)
  expl_off     explosive plays = pass >= 20 yds or run >= 10 yds, per offensive play
  expl20_off   any scrimmage play >= 20 yds, per play
  expl_def / expl20_def   the same rates ALLOWED by the team's defense (opponent's offense)
  epa_off / epa_def, succ_off / succ_def   EPA per play and success rate (epa>0), for / against
  def_plays    plays run by the opponent's offense (defensive snaps faced)
Output: experts/pace/_teamgame.csv (one row per gid x team). Re-runnable; prints coverage.
"""
import sys, time
import numpy as np, pandas as pd
import statsmodels.api as sm
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kit import load_games, norm

SCRAPR = "/home/user/nflscrapR-data/play_by_play_data/regular_season/reg_pbp_{}.csv"
FASTR = "/home/user/NFLkz/data/cache/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "home_team", "away_team", "posteam", "defteam", "play_type", "yards_gained", "yardline_100",
        "down", "ydstogo", "qtr", "game_seconds_remaining", "half_seconds_remaining", "score_differential",
        "posteam_score", "posteam_score_post", "drive", "no_huddle", "wp", "epa", "play_id"]

g = load_games(min_season=2009)
old2gid = dict(zip(g.old_game_id.astype("Int64").astype(str), g.gid))
new2gid = dict(zip(g.game_id, g.gid))


def load_season(yr):
    if yr <= 2019:
        d = pd.read_csv(SCRAPR.format(yr), usecols=COLS, low_memory=False)
        d["gid"] = d.game_id.astype(str).map(old2gid)
        d["pass_oe"] = np.nan
    else:
        d = pd.read_parquet(FASTR.format(yr), columns=COLS + ["fixed_drive", "pass_oe"])
        d["drive"] = d.fixed_drive
        d["gid"] = d.game_id.map(new2gid)
    d["season"] = yr
    for c in ("posteam", "defteam"):
        d[c] = d[c].map(lambda t: norm(t) if isinstance(t, str) else t)
    unm = d.gid.isna().mean()
    d = d[d.gid.notna() & d.posteam.notna() & d.defteam.notna()].copy()
    d = d.sort_values(["gid", "play_id"]).reset_index(drop=True)
    d["scrim"] = d.play_type.isin(["pass", "run"])
    d["is_pass"] = (d.play_type == "pass").astype(float)
    d["neutral"] = ((d.qtr <= 3) & (d.score_differential.abs() <= 8)
                    & ~((d.qtr == 2) & (d.half_seconds_remaining <= 120))).astype(int)
    return d, unm


def xpass_design(s):
    """design matrix for the expected-pass logit (scrimmage plays only)."""
    X = pd.DataFrame(index=s.index)
    dn = s.down.fillna(1).clip(1, 4)
    for k in (2, 3, 4):
        X[f"down{k}"] = (dn == k).astype(float)
    ytg = s.ydstogo.clip(1, 30).astype(float)
    X["ytg"] = ytg; X["log_ytg"] = np.log(ytg)
    for k in (2, 3):
        X[f"down{k}_ytg"] = X[f"down{k}"] * ytg
    X["gtg"] = (s.yardline_100 <= s.ydstogo).astype(float)
    yl = s.yardline_100.astype(float)
    X["yl"] = yl; X["yl_sq"] = yl ** 2 / 100.0; X["rz"] = (yl <= 20).astype(float)
    sd = s.score_differential.clip(-28, 28).astype(float)
    gfrac = 1 - s.game_seconds_remaining.clip(0, 3600) / 3600.0        # 0 at kickoff, 1 at end
    X["sd"] = sd; X["sd_late"] = sd * gfrac; X["sd_sq_late"] = np.sign(sd) * sd ** 2 / 28.0 * gfrac
    X["gfrac"] = gfrac; X["hsr"] = s.half_seconds_remaining.clip(0, 1800) / 1800.0
    X["h2"] = (s.qtr >= 3).astype(float); X["q4"] = (s.qtr >= 4).astype(float)
    X["two_min"] = ((s.half_seconds_remaining <= 120)).astype(float)
    X["two_min_sd"] = X.two_min * sd
    wp = s.wp.clip(0.01, 0.99).fillna(0.5).astype(float)
    X["wp"] = wp; X["wp_sq"] = wp ** 2; X["wp_late"] = wp * gfrac
    return sm.add_constant(X, has_constant="add")


t0 = time.time()
seasons = list(range(2009, 2020)) + [2023, 2024, 2025]
data = {}
for yr in seasons:
    d, unm = load_season(yr)
    data[yr] = d
    print(f"loaded {yr}: rows={len(d)} games={d.gid.nunique()} unmapped_rows={unm:.4f}", flush=True)

# ---------------- expected-pass model, fit on 2009-2019 only ----------------
fit_rows = pd.concat([data[y][data[y].scrim] for y in range(2009, 2020)], ignore_index=True)
Xf = xpass_design(fit_rows); yf = fit_rows.is_pass.values
ok = Xf.notna().all(axis=1).values
logit = sm.Logit(yf[ok], Xf[ok].astype(float)).fit(disp=0, maxiter=200)
print(f"\nxpass logit fit on 2009-2019 scrimmage plays n={ok.sum()}  pseudo-R2={logit.prsquared:.3f}")
print(logit.params.round(3).to_string())


def team_game(d):
    s = d[d.scrim].copy()
    X = xpass_design(s)
    s["xpass"] = logit.predict(X.astype(float))
    s["poe"] = (s.is_pass - s.xpass) * 100.0
    s["expl"] = (((s.play_type == "pass") & (s.yards_gained >= 20)) | ((s.play_type == "run") & (s.yards_gained >= 10))).astype(int)
    s["expl20"] = (s.yards_gained >= 20).astype(int)
    s["succ"] = (s.epa > 0).astype(int)
    s["wp_neut"] = ((s.qtr <= 3) & s.wp.between(0.2, 0.8)).astype(int)
    # pace: gaps between consecutive offensive snaps within the same drive
    grp = s.groupby(["gid", "drive", "posteam"], sort=False)
    dt = grp.game_seconds_remaining.shift(1) - s.game_seconds_remaining
    prev_run = grp.play_type.shift(1).eq("run")
    okdt = dt.notna() & (dt > 0) & (dt <= 60) & (s.neutral == 1)
    s["gap"] = np.where(okdt, dt, np.nan)
    s["gap_run"] = np.where(okdt & prev_run, dt, np.nan)
    nt = s[s.neutral == 1]
    wn = s[s.wp_neut == 1]
    off = s.groupby(["gid", "posteam"]).agg(
        plays=("play_id", "size"), pr_all=("is_pass", "mean"), proe=("poe", "mean"),
        proe_fastr=("pass_oe", "mean"), expl_off=("expl", "mean"), expl20_off=("expl20", "mean"),
        expl_off_n=("expl", "sum"), epa_off=("epa", "mean"), succ_off=("succ", "mean"),
        spp_neut=("gap", "mean"), spp_neut_n=("gap", "count"), spp_neut_run=("gap_run", "mean"),
        spp_neut_run_n=("gap_run", "count")).reset_index()
    offn = nt.groupby(["gid", "posteam"]).agg(plays_neut=("play_id", "size"), pr_neut=("is_pass", "mean"),
                                              nh_neut=("no_huddle", "mean"), proe_neut=("poe", "mean")).reset_index()
    offw = wn.groupby(["gid", "posteam"]).agg(pr_wp=("is_pass", "mean")).reset_index()
    dff = s.groupby(["gid", "defteam"]).agg(def_plays=("play_id", "size"), expl_def=("expl", "mean"),
                                           expl20_def=("expl20", "mean"), epa_def=("epa", "mean"),
                                           succ_def=("succ", "mean")).reset_index().rename(columns={"defteam": "posteam"})
    # drives & points per drive (all offensive plays except kickoffs / XPs / empty)
    dd = d[~d.play_type.isin(["kickoff"]) & d.play_type.notna() & d.drive.notna()]   # XP rows kept so TD drives count 7
    dr = dd.groupby(["gid", "posteam", "drive"]).agg(s0=("posteam_score", "min"), s1=("posteam_score_post", "max")).reset_index()
    dr["pts"] = (dr.s1 - dr.s0).clip(lower=0)
    # drives = distinct drive ids with at least one pass/run/punt/FG/kneel/spike (XP-only rows do not add drives)
    ndr = dd[~dd.play_type.isin(["extra_point"])].groupby(["gid", "posteam"]).drive.nunique().rename("drives").reset_index()
    drv = dr.groupby(["gid", "posteam"]).agg(pts_off=("pts", "sum")).reset_index().merge(ndr, on=["gid", "posteam"], how="left")
    out = off.merge(offn, on=["gid", "posteam"], how="left").merge(offw, on=["gid", "posteam"], how="left")
    out = out.merge(dff, on=["gid", "posteam"], how="left").merge(drv, on=["gid", "posteam"], how="left")
    out["ppd"] = out.pts_off / out.drives
    return out.rename(columns={"posteam": "team"})


frames = []
for yr in seasons:
    tg = team_game(data[yr]); tg["season"] = yr
    frames.append(tg)
    print(f"{yr}: team-games={len(tg)}  plays={tg.plays.mean():.1f} spp_neut={tg.spp_neut.mean():.1f} pr_neut={tg.pr_neut.mean():.3f} "
          f"expl_off={tg.expl_off.mean():.3f} ppd={tg.ppd.mean():.2f} proe={tg.proe.mean():+.2f}", flush=True)
tg = pd.concat(frames, ignore_index=True)
# attach schedule info
long = pd.concat([g[["gid", "season", "week", "gameday", "game_type", "home", "away", "home_score", "away_score"]]
                  .rename(columns={"home": "team", "away": "opp", "home_score": "pts", "away_score": "pts_allowed"}).assign(is_home=1),
                  g[["gid", "season", "week", "gameday", "game_type", "away", "home", "away_score", "home_score"]]
                  .rename(columns={"away": "team", "home": "opp", "away_score": "pts", "home_score": "pts_allowed"}).assign(is_home=0)])
tg = long.merge(tg.drop(columns=["season"]), on=["gid", "team"], how="inner")
tg = tg.sort_values(["season", "gameday", "gid", "is_home"]).reset_index(drop=True)
tg.to_csv(HERE / "_teamgame.csv", index=False)
print(f"\nwrote {HERE/'_teamgame.csv'} shape={tg.shape} in {time.time()-t0:.0f}s")
print("coverage by season (team-games):", tg.groupby("season").size().to_dict())
print("\nsanity: corr(pts_off, pts) =", round(tg[["pts_off", "pts"]].corr().iloc[0, 1], 3), "(offensive drive points vs team score; should be ~0.95)")
print("sanity: mean (pts - pts_off) =", round((tg.pts - tg.pts_off).mean(), 2), "(defensive/ST return points per game; should be ~0.5-1.5)")
f = tg[tg.season >= 2023]
print("proe vs nflfastR pass_oe (2023-25 team-games): corr =", round(f[["proe", "proe_fastr"]].corr().iloc[0, 1], 3),
      "| mean proe", round(f.proe.mean(), 2), "mean pass_oe", round(f.proe_fastr.mean(), 2))
print(tg[["plays", "drives", "ppd", "spp_neut", "spp_neut_run", "spp_neut_n", "nh_neut", "pr_neut", "pr_all", "pr_wp", "proe", "proe_neut",
          "expl_off", "expl20_off", "expl_def", "epa_off", "epa_def"]].describe().round(3).T.to_string())
