"""04 / THEORY 4: new head coach / new starting QB in Week 1 — residual vs the market and vs the rating line.
Flags built from games.csv (1999-2025, REG):
  new_hc   : team's Week-1 head coach did not coach the team in ANY game of the previous season (offseason change)
  hc_first : new_hc AND the coach never appears as a head coach in games.csv before this season (seasons >= 2002 only,
             so at least 3 seasons of look-back exist; before that the flag is unknown)
  new_qb   : Week-1 starter != previous season's primary starter (most REG starts for that team)
  qb_new_to_team : Week-1 starter made no REG start for the team in the previous season
Game-level regressions: err (home perspective) ~ net flag (home flag - away flag), HC1 se.  Also the market's pricing of the
flag relative to the rating line (mkt - elo_line), and residual vs elo_line / nraw (2009+).
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/earlyseason")
import numpy as np, pandas as pd
from scipy import stats
from common import build, desc, binom, ols, paired_mae

pd.set_option("display.width", 250)
m = build(min_season=1999)
# --- team-season table -------------------------------------------------------------------------
h = m[["season", "week", "home", "home_coach", "home_qb_name"]].rename(columns={"home": "team", "home_coach": "coach", "home_qb_name": "qb"})
a = m[["season", "week", "away", "away_coach", "away_qb_name"]].rename(columns={"away": "team", "away_coach": "coach", "away_qb_name": "qb"})
tg = pd.concat([h, a]).sort_values(["team", "season", "week"])
rows = []
for (team, season), x in tg.groupby(["team", "season"]):
    prev = tg[(tg.team == team) & (tg.season == season - 1)]
    w1 = x[x.week == x.week.min()]
    if len(w1) == 0: continue
    coach1, qb1 = w1.coach.iloc[0], w1.qb.iloc[0]
    prev_coaches = set(prev.coach.dropna()); prev_qbs = set(prev.qb.dropna())
    prev_primary_qb = prev.qb.mode().iloc[0] if len(prev) else np.nan
    prev_primary_coach = prev.coach.mode().iloc[0] if len(prev) else np.nan
    seen_before = set(tg[(tg.season < season)].coach.dropna())
    rows.append(dict(team=team, season=season, w1_week=w1.week.iloc[0], coach=coach1, qb=qb1,
                     new_hc=int(len(prev) > 0 and coach1 not in prev_coaches),
                     hc_first=int(len(prev) > 0 and coach1 not in prev_coaches and coach1 not in seen_before) if season >= 2002 else np.nan,
                     new_qb=int(len(prev) > 0 and qb1 != prev_primary_qb),
                     qb_new_to_team=int(len(prev) > 0 and qb1 not in prev_qbs),
                     has_prev=int(len(prev) > 0)))
ts = pd.DataFrame(rows)
ts = ts[ts.has_prev == 1]
print("team-seasons with a previous season: %d (2000-2025) | new_hc %d (%.1f/season) | hc_first %d | new_qb %d | qb_new_to_team %d" %
      (len(ts), ts.new_hc.sum(), ts.new_hc.sum() / ts.season.nunique(), ts.hc_first.sum(), ts.new_qb.sum(), ts.qb_new_to_team.sum()))
print("  2026 Week-1 flags (from the schedule file, for the current card):")
g26 = pd.read_csv("/home/user/originator-2026-w01/research/data/games_1999_2025.csv", low_memory=False)
g26 = g26[(g26.season == 2026) & (g26.week == 1)]
prev25 = tg[tg.season == 2025]
for _, r in g26.iterrows():
    for side in ["home", "away"]:
        t = r[f"{side}_team"]; t = {"LAR": "LA", "OAK": "LV"}.get(t, t); c = r[f"{side}_coach"]; q = r[f"{side}_qb_name"]
        pc = set(prev25[prev25.team == t].coach.dropna()); pq = prev25[prev25.team == t].qb.mode()
        flag_hc = (c not in pc) if isinstance(c, str) else None; flag_qb = (q != (pq.iloc[0] if len(pq) else None)) if isinstance(q, str) else None
        if flag_hc or flag_qb:
            print("    %s %-4s coach=%s new_hc=%s | qb=%s new_qb=%s (2025 primary: %s)" % (r.game_id, t, c, flag_hc, q, flag_qb, pq.iloc[0] if len(pq) else "?"))

# --- merge flags onto games --------------------------------------------------------------------
for side in ["home", "away"]:
    m = m.merge(ts[["team", "season", "new_hc", "hc_first", "new_qb", "qb_new_to_team"]].rename(columns={"team": side, **{c: f"{side}_{c}" for c in ["new_hc", "hc_first", "new_qb", "qb_new_to_team"]}}),
                on=[side, "season"], how="left")
for f in ["new_hc", "hc_first", "new_qb", "qb_new_to_team"]:
    m[f"net_{f}"] = m[f"home_{f}"] - m[f"away_{f}"]
m["net_hc_and_qb"] = (m.home_new_hc * m.home_new_qb) - (m.away_new_hc * m.away_new_qb)
m["mkt_minus_elo"] = m.mkt - m.elo_line          # + = market likes home LESS than ratings do (line more positive)
w1 = m[(m.week == 1) & m.home_new_hc.notna()].copy()


def run(y, x, df, label):
    dd = df.dropna(subset=[y, x])
    ev = int((dd[x] != 0).sum())
    if ev < 5: print("  %-52s n=%4d events=%3d  (too few)" % (label, len(dd), ev)); return
    co, r = ols(dd[y], [dd[x]], ["b"])
    b, se, p = co["b"]
    print("  %-52s n=%4d events=%3d coef=%+.3f se=%.3f 95%%CI [%+.2f,%+.2f] p=%.3f" % (label, len(dd), ev, b, se, b - 1.96 * se, b + 1.96 * se, p))


def ats_back(df, x, label):
    """ATS record of BACKING the flagged team vs the market."""
    dd = df[df[x].notna() & (df[x] != 0) & df.err_mkt.notna()]
    win = ((dd[x] > 0) & (dd.err_mkt > 0)) | ((dd[x] < 0) & (dd.err_mkt < 0)); push = dd.err_mkt == 0
    w = int(win.sum()); l = int((~win & ~push).sum()); pct, lo, hi, p = binom(w, l)
    print("  %-52s back-flag ATS %d-%d-%d = %.3f [%.3f,%.3f] p=%.3f" % (label, w, l, int(push.sum()), pct, lo, hi, p))


print("\nA. WEEK 1, residual vs MARKET close (1999-2025; positive coef = flagged team beat the market by that many points)")
print("   Week-1 games with flags: %d | home/away team-games new_hc=%d new_qb=%d hc_first=%d qb_new_to_team=%d" %
      (len(w1), int(w1.home_new_hc.sum() + w1.away_new_hc.sum()), int(w1.home_new_qb.sum() + w1.away_new_qb.sum()), int(w1.home_hc_first.sum() + w1.away_hc_first.sum()), int(w1.home_qb_new_to_team.sum() + w1.away_qb_new_to_team.sum())))
for f, lab in [("new_hc", "new head coach"), ("hc_first", "first-time head coach"), ("new_qb", "new starting QB (vs prev primary)"), ("qb_new_to_team", "QB with no start for team last season"), ("hc_and_qb", "new HC AND new QB")]:
    run("err_mkt", f"net_{f}", w1, f"{lab}: resid vs market ALL")
    run("err_mkt", f"net_{f}", w1[w1.fit], f"{lab}: resid vs market FIT <=2021")
    run("err_mkt", f"net_{f}", w1[w1.test], f"{lab}: resid vs market TEST 2022-25")
    ats_back(w1, f"net_{f}", f"{lab}: ")
# joint
dd = w1.dropna(subset=["net_new_hc", "net_new_qb"])
co, r = ols(dd.err_mkt, [dd.net_new_hc, dd.net_new_qb], ["hc", "qb"])
print("  joint W1: new_hc %+.2f (se %.2f p=%.3f) | new_qb %+.2f (se %.2f p=%.3f) n=%d" % (co["hc"][0], co["hc"][1], co["hc"][2], co["qb"][0], co["qb"][1], co["qb"][2], len(dd)))

print("\nB. WEEK 1, 2009+ with ratings: does the market price the flag, and is the residual vs the RATING line different?")
w9 = w1[w1.elo_line.notna()]
for f, lab in [("new_hc", "new head coach"), ("new_qb", "new starting QB")]:
    run("mkt_minus_elo", f"net_{f}", w9, f"{lab}: market line minus rating line (pricing)")
    run("err_elo_line", f"net_{f}", w9, f"{lab}: resid vs rating line (elo+HFA, no QB adj)")
    run("err_nraw", f"net_{f}", w9, f"{lab}: resid vs nfelo raw (with 538 QB adj)")
    run("err_mkt", f"net_{f}", w9, f"{lab}: resid vs market (same sample)")

print("\nC. Persistence: weeks 1-4 and full season, residual vs market (flag is a season-long attribute)")
for wk_lab, mask in [("W1-4", m.week.between(1, 4)), ("W5-18", m.week >= 5), ("ALL weeks", m.week >= 1)]:
    x = m[mask & m.home_new_hc.notna()]
    for f, lab in [("new_hc", "new HC"), ("hc_first", "first-time HC"), ("new_qb", "new QB")]:
        run("err_mkt", f"net_{f}", x, f"{wk_lab} {lab}: resid vs market ALL")
        run("err_mkt", f"net_{f}", x[x.test], f"{wk_lab} {lab}: resid vs market TEST")
        ats_back(x, f"net_{f}", f"{wk_lab} {lab}")

print("\nD. Direct team-level view, Week 1: mean margin-vs-line for flagged teams (team perspective) with CI")
def team_view(df, flag, label):
    vals = pd.concat([df[df[f"home_{flag}"] == 1].err_mkt, -df[df[f"away_{flag}"] == 1].err_mkt])
    n, mu, se, p = desc(vals); w = int((vals > 0).sum()); l = int((vals < 0).sum())
    print("  %-38s team-games n=%3d  mean cover margin %+.2f (se %.2f) p=%.3f | ATS %d-%d %.3f" % (label, n, mu, se, p, w, l, w / max(w + l, 1)))
for flag, lab in [("new_hc", "new HC (W1)"), ("hc_first", "first-time HC (W1)"), ("new_qb", "new QB (W1)"), ("qb_new_to_team", "QB new to team (W1)")]:
    team_view(w1, flag, lab)
    team_view(w1[w1.test], flag, lab + " TEST")
print("  new HC by era (W1):")
for lab, mask in [("1999-2008", w1.season <= 2008), ("2009-2016", w1.season.between(2009, 2016)), ("2017-2025", w1.season >= 2017)]:
    team_view(w1[mask], "new_hc", "  new HC " + lab)
