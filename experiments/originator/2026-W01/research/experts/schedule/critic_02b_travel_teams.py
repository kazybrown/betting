"""CRITIC 02b - T4b follow-up: is the 2018-25 3-zone 'traveller edge' a travel effect or a franchise effect?
Compare SEA/SF (as away teams) and LA/LAC/LV (as hosts) in their 3-zone games vs their OTHER games (vs market and vs ratings);
bootstrap CI on the rolling-origin pooled MAE change from critic_02 (b7); and check the sign of home_line_open in the nfelo file."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_nfelo
_n = load_nfelo()[["gid", "home_line_open"]].drop_duplicates("gid")
from common import build, desc, ats_side
pd.set_option("display.width", 250); rng = np.random.default_rng(11)
m = build().merge(_n, on='gid', how='left'); m = m[~m.neutral].copy(); d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["xc"] = d.tz_diff.abs() == 3; x = d[d.season >= 2018]
print("=== franchise vs travel, 2018-25 (home-perspective residual vs market; for away-team rows the sign is flipped to the team's perspective) ===")
for lab, team_mask, persp in [("SEA/SF as AWAY", x.away_team.isin(["SEA", "SF"]), -1), ("LA/LAC/LV as HOME", x.home_team.isin(["LA", "LAR", "LAC", "LV", "OAK"]), 1),
                              ("other west (ARI/LA/LAC/LV) as AWAY", x.away_team.isin(["ARI", "LA", "LAR", "LAC", "LV", "OAK"]), -1), ("ET teams as AWAY", x.away_off == 0, -1), ("ET teams as HOME", x.home_off == 0, 1)]:
    for sub, mk in [("3-zone games", x.xc), ("non-3-zone games", ~x.xc)]:
        y = x[team_mask & mk]; n, mu, se, p = desc(y.err_mkt * persp); n2, mu2, se2, p2 = desc(y.err_rate * persp)
        print("  %-34s %-16s n=%3d team vs market %+.2f (se %.2f p=%.2f) | vs ratings %+.2f (se %.2f)" % (lab, sub, n, mu, se, p, mu2, se2))
print("\n=== residual of the 3-zone effect after removing each side's own average residual in the season (team-season demeaned, 2018-25) ===")
# demean err_mkt by home-team-season and away-team-season means computed on NON-xc games only (so the xc games don't define the baseline)
base_h = x[~x.xc].groupby(["season", "home_team"]).err_mkt.mean().rename("h_base"); base_a = x[~x.xc].groupby(["season", "away_team"]).err_mkt.mean().rename("a_base")
xx = x[x.xc].join(base_h, on=["season", "home_team"]).join(base_a, on=["season", "away_team"])
xx["adj"] = xx.err_mkt - xx.h_base.fillna(0) - xx.a_base.fillna(0)
n, mu, se, p = desc(xx.adj); print("  3-zone games 2018-25 n=%d: raw err_mkt %+.2f -> demeaned by host & traveller season baselines %+.2f (se %.2f p=%.3f)" % (n, xx.err_mkt.mean(), mu, se, p))
print("\n=== bootstrap CI for the rolling-origin pooled MAE change (critic_02 b7 design, seasons 2014-25) ===")
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["c1"] = 0.46 * (d.nraw_line + d.nfelo_bye_pts + d.nfelo_tz_pts) + 0.54 * d.rate_line - 0.15 * d.rd
parts = []
for s in range(2014, 2026):
    f = d[(d.season < s) & d.xc]; t = d[(d.season == s) & d.xc]; ks = np.arange(-1, 4.01, 0.25)
    k = ks[int(np.argmin([(f.margin + f.rate_line + kk).abs().mean() for kk in ks]))]
    e0 = (t.margin + t.c1).abs().values; e1 = (t.margin + t.c1 + k).abs().values; parts.append(e1 - e0)
diff = np.concatenate(parts); bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
print("  n=%d pooled MAE change %+.4f, 95%% CI [%+.4f, %+.4f]" % (len(diff), diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
parts22 = np.concatenate(parts[-4:]); bs2 = [rng.choice(parts22, len(parts22)).mean() for _ in range(4000)]
print("  2022-25 only n=%d: %+.4f, CI [%+.4f, %+.4f]" % (len(parts22), parts22.mean(), np.percentile(bs2, 2.5), np.percentile(bs2, 97.5)))
print("\n=== sign of home_line_open (nfelo file) ===")
o = d.dropna(subset=["home_line_open"])
print("  corr(home_line_open, margin) = %.3f | corr(home_line_open, mkt_spread[= -spread_line]) = %.3f | mean |open - close| = %.2f | n=%d" %
      (np.corrcoef(o.home_line_open, o.margin)[0, 1], np.corrcoef(o.home_line_open, o.mkt_spread)[0, 1], (o.home_line_open - o.mkt_spread).abs().mean(), len(o)))
