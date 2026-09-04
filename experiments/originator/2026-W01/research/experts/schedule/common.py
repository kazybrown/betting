"""Shared frame for the schedule expert (rest / travel / week-of-season).

Residual series (all home perspective, POSITIVE = home did better than the line):
  err_mkt    = margin + mkt_spread                       market closing line
  err_nclose = margin + nfelo_home_line_close            nfelo published close (market-regressed, ~0.7 toward market)
  err_nraw   = margin - nfelo_dif_base/25                nfelo UNREGRESSED line: ratings + HFA + QB + bye + tz + div + surface
  err_rate   = margin - (elo dif + hfa_base + qb + div + surface)/25  'rating-only' line: NO bye / NO time-zone mods
                                                         (this is what the PFF / Cole paths in ORIGINATOR look like)
Conventions: ORIGINATOR spread negative = home favored; pred error = margin + pred.
"""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from kit import load_games, load_nfelo

FIT_MAX = 2021   # fit seasons <= 2021, test 2022-2025

# --- team home time zone offsets vs Eastern (hours; negative = west). Raw nflverse ids (STL/SD/OAK kept).
TZ = {**{t: 0 for t in ["BUF","MIA","NE","NYJ","NYG","BAL","CIN","CLE","PIT","JAX","IND","ATL","CAR","PHI","WAS","DET","TB"]},
      **{t: -1 for t in ["HOU","TEN","KC","CHI","GB","MIN","DAL","NO","STL"]},
      "DEN": -2, "ARI": -2,   # ARI: no DST -> effectively Pacific until early Nov (handled in tz_offset)
      **{t: -3 for t in ["LA","LAR","LAC","SD","OAK","LV","SEA","SF"]}}
# --- approximate stadium coordinates (lat, lon) for great-circle travel distance
LOC = {"ARI": (33.53,-112.26), "ATL": (33.76,-84.40), "BAL": (39.28,-76.62), "BUF": (42.77,-78.79), "CAR": (35.23,-80.85),
       "CHI": (41.86,-87.62), "CIN": (39.10,-84.52), "CLE": (41.51,-81.70), "DAL": (32.75,-97.09), "DEN": (39.74,-105.02),
       "DET": (42.34,-83.05), "GB": (44.50,-88.06), "HOU": (29.68,-95.41), "IND": (39.76,-86.16), "JAX": (30.32,-81.64),
       "KC": (39.05,-94.48), "LA": (33.95,-118.34), "LAR": (33.95,-118.34), "LAC": (33.95,-118.34), "LV": (36.09,-115.18),
       "MIA": (25.96,-80.24), "MIN": (44.97,-93.26), "NE": (42.09,-71.26), "NO": (29.95,-90.08), "NYG": (40.81,-74.07),
       "NYJ": (40.81,-74.07), "PHI": (39.90,-75.17), "PIT": (40.45,-80.02), "SEA": (47.60,-122.33), "SF": (37.40,-121.97),
       "TB": (27.98,-82.50), "TEN": (36.17,-86.77), "WAS": (38.91,-76.86), "STL": (38.63,-90.19), "SD": (32.78,-117.12),
       "OAK": (37.75,-122.20)}

def tz_offset(team, gameday):
    off = TZ[team]
    if team == "ARI":   # Phoenix stays UTC-7 all year: = Pacific clock while the rest of the US is on DST
        d = pd.Timestamp(gameday)
        first_sun_nov = pd.Timestamp(d.year, 11, 1) + pd.Timedelta(days=(6 - pd.Timestamp(d.year, 11, 1).weekday()) % 7)
        off = -3 if d < first_sun_nov else -2
    return off

def haversine(a, b):
    la1, lo1 = np.radians(a); la2, lo2 = np.radians(b)
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))

def build(reg_only=True):
    g = load_games()
    n = load_nfelo()
    keep = ["gid","starting_nfelo_home","starting_nfelo_away","hfa_mod","hfa_base_mod","home_net_qb_mod","div_game_mod",
            "dif_surface_mod","home_time_advantage_mod","home_bye_mod","away_bye_mod","home_net_bye_mod","nfelo_dif_base",
            "nfelo_home_line_close","nfelo_home_line_open","home_line_close","total_line_close"]
    m = g.merge(n[[k for k in keep if k in n.columns]], on="gid", how="left")
    if reg_only: m = m[m.game_type.eq("REG")].copy()
    m = m.dropna(subset=["mkt_spread"]).copy()
    m["fit"] = m.season <= FIT_MAX
    m["test"] = m.season >= FIT_MAX + 1
    # residual series
    m["err_mkt"] = m.margin + m.mkt_spread
    m["err_nclose"] = m.margin + m.nfelo_home_line_close
    m["nraw_line"] = -m.nfelo_dif_base / 25.0
    m["err_nraw"] = m.margin + m.nraw_line
    # nfelo_dif_base == elo dif + hfa_base_mod + tz + qb + div + surface + home_bye + away_bye (verified 00b, resid sd 3 Elo)
    m["nfelo_bye_pts"] = (m.home_bye_mod.fillna(0) + m.away_bye_mod.fillna(0)) / 25.0   # + = favors home
    m["nfelo_tz_pts"] = m.home_time_advantage_mod.fillna(0) / 25.0
    m["rate_line"] = m.nraw_line + m.nfelo_bye_pts + m.nfelo_tz_pts     # strip nfelo's rest + time-zone mods
    m["err_rate"] = m.margin + m.rate_line
    m["tot_err_mkt"] = m.total_pts - m.mkt_total
    # rest features (week 1 rest is a placeholder = 7 -> flag)
    m["rest_valid"] = m.week > 1
    m["home_short"] = (m.home_rest <= 5) & m.rest_valid
    m["away_short"] = (m.away_rest <= 5) & m.rest_valid
    m["home_bye"] = (m.home_rest >= 13) & m.rest_valid
    m["away_bye"] = (m.away_rest >= 13) & m.rest_valid
    m["home_mini"] = (m.home_rest == 6) & m.rest_valid    # e.g. Monday -> Sunday
    m["away_mini"] = (m.away_rest == 6) & m.rest_valid
    m["home_long"] = m.home_rest.between(9, 12) & m.rest_valid   # e.g. Thursday -> Sunday (10 days)
    m["away_long"] = m.away_rest.between(9, 12) & m.rest_valid
    m["rest_diff"] = np.where(m.rest_valid, m.home_rest - m.away_rest, 0)
    # kickoff / time zones (gametime is Eastern)
    m["kick_et"] = m.gametime.str.slice(0, 2).astype(int) + m.gametime.str.slice(3, 5).astype(int) / 60.0
    m["home_off"] = [tz_offset(t, d) for t, d in zip(m.home_team, m.gameday)]
    m["away_off"] = [tz_offset(t, d) for t, d in zip(m.away_team, m.gameday)]
    m["home_body"] = m.kick_et + m.home_off
    m["away_body"] = m.kick_et + m.away_off
    m["tz_diff"] = m.home_off - m.away_off        # >0: home is east of away (away travels east)
    m["dist"] = [haversine(LOC[a], LOC[h]) for a, h in zip(m.away_team, m.home_team)]
    m["primetime"] = m.kick_et >= 19.0
    m["early"] = m.kick_et.between(12.5, 13.5)   # 1pm ET window (12:30 - 13:30)
    m["late"] = m.kick_et.between(16.0, 16.6)    # 4pm ET window
    return m

def desc(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); mu = x.mean() if n else np.nan; se = x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
    p = stats.ttest_1samp(x, 0).pvalue if n > 2 else np.nan
    return n, mu, se, p

def mean_row(label, d, cols=("err_mkt","err_nclose","err_nraw","err_rate")):
    out = [label, len(d)]
    for c in cols:
        n, mu, se, p = desc(d[c]); out += ["%+.2f (se %.2f, p=%.2f)" % (mu, se, p)]
    return out

def report_means(title, d, groups, cols=("err_mkt","err_nclose","err_nraw","err_rate")):
    """groups: list of (label, boolean mask). Prints mean residual by group for each residual series."""
    print("\n" + title)
    print("  %-38s %5s | %-26s %-26s %-26s %-26s" % ("group", "n", *[c for c in cols]))
    for lab, mask in groups:
        x = d[mask]
        cells = []
        for c in cols:
            n, mu, se, p = desc(x[c]); cells.append("%+.2f (se %.2f p=%.2f)" % (mu, se, p) if n > 2 else "n/a")
        print("  %-38s %5d | %s" % (lab, len(x), " ".join("%-26s" % c for c in cells)))

def ols(y, X, names, cov="HC1"):
    X = sm.add_constant(np.column_stack(X)) if len(names) > 1 else sm.add_constant(np.asarray(X[0], float))
    r = sm.OLS(np.asarray(y, float), X, missing="drop").fit(cov_type=cov)
    return {nm: (r.params[i+1], r.bse[i+1], r.pvalues[i+1]) for i, nm in enumerate(names)}, r

def ats_side(err_mkt, side_sign):
    """ATS record of betting the side indicated by side_sign (+1 = home, -1 = away) vs the closing line.
    err_mkt = margin + mkt_spread (>0 home covers)."""
    e = np.asarray(err_mkt) * np.asarray(side_sign)
    w = int((e > 0).sum()); l = int((e < 0).sum()); p = int((e == 0).sum())
    pv = stats.binomtest(w, w + l, 0.5).pvalue if w + l > 0 else np.nan
    return w, l, p, (w / (w + l) if w + l else np.nan), pv

if __name__ == "__main__":
    m = build()
    print("REG games with market line:", len(m), "| fit:", m.fit.sum(), "| test:", m.test.sum())
    print("nfelo coverage:", m.nfelo_dif_base.notna().mean().round(3), "| test coverage:", m[m.test].nfelo_dif_base.notna().mean())
    d = m.dropna(subset=["nfelo_dif_base"])
    print("sanity corr(mkt_spread, margin) = %.3f (should be strongly negative)" % np.corrcoef(d.mkt_spread, d.margin)[0,1])
    print("MAE vs results: market %.3f | nfelo close %.3f | nfelo raw %.3f | rating-only %.3f" %
          (d.err_mkt.abs().mean(), d.err_nclose.abs().mean(), d.err_nraw.abs().mean(), d.err_rate.abs().mean()))
    t = d[d.test]
    print("TEST 2022-25 MAE: market %.3f | nfelo close %.3f | nfelo raw %.3f | rating-only %.3f (n=%d)" %
          (t.err_mkt.abs().mean(), t.err_nclose.abs().mean(), t.err_nraw.abs().mean(), t.err_rate.abs().mean(), len(t)))
    print("check: nraw_line - rate_line should equal -(bye+tz)/25:  max abs dev = %.4f" %
          ((d.nraw_line - d.rate_line) + d.nfelo_bye_pts + d.nfelo_tz_pts).abs().max())
    print("corr(nraw_line, mkt_spread)=%.3f  corr(nclose, mkt)=%.3f" % (np.corrcoef(d.nraw_line, d.mkt_spread)[0,1], np.corrcoef(d.nfelo_home_line_close, d.mkt_spread)[0,1]))
    print("tz_diff counts:", m.tz_diff.value_counts().sort_index().to_dict())
    print("home_body kickoff hours (top):", m.home_body.round(2).value_counts().head(8).to_dict())
    print("away_body kickoff hours (top):", m.away_body.round(2).value_counts().head(8).to_dict())
    print("dist describe:", m.dist.describe().round(0).to_dict())
    print("example ARI DST check:", m[m.home_team.eq("ARI")][["gameday","home_off"]].drop_duplicates("home_off").head(4).to_dict("records"))
