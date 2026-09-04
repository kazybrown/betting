"""CRITIC 06 (T2 bound): how much does a TRULY market-blind rating lose vs the close?
nfelo's source (greerreNFL/nfelo config.json v4.3.1, Utilities/elo_shift.py) shows its rating update is market-informed:
  calc_shift inflates k by (1 + |model_error - market_error| / market_resist_factor[=1.5039]) whenever the model missed
  by more than the closing line did, and the 'observations' blend margin 0.7382 / PFF point margin 0.1113 / WEPA 0.1506.
So nfelo_lin is not a market-blind number. Here we build a plain results-only Elo with the canonical 538 parameters
(K=20, log MOV multiplier, 1/3 offseason reversion to 1505; nothing fitted on this data), give it nfelo's own per-game HFA
(hfa_mod) so the only difference from nfelo_noqb is the rating series, and score it the same way.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import load, paired_mae_test, ROOT
import sys; sys.path.insert(0, str(ROOT))
from kit import load_games
pd.set_option("display.width", 220)
m = load(verbose=False)

def run_elo(K=20, rev=1/3, hfa_default_pts=2.2):
    g = load_games(min_season=1999).sort_values(["season", "gameday", "game_id"]).copy()
    hfa = m.set_index("gid").hfa_pts  # nfelo per-game HFA in points (2009+)
    g["hfa_pts_use"] = g.gid.map(hfa).fillna(hfa_default_pts)
    g.loc[g.neutral, "hfa_pts_use"] = 0.0
    elo = {}; cur_season = None; pre_h = []; pre_a = []
    for r in g.itertuples(index=False):
        if r.season != cur_season:
            for t in elo: elo[t] = 1505 + (elo[t] - 1505) * (1 - rev)
            cur_season = r.season
        eh, ea = elo.get(r.home, 1505.0), elo.get(r.away, 1505.0)
        pre_h.append(eh); pre_a.append(ea)
        dif = eh - ea + 25 * r.hfa_pts_use
        p = 1 / (1 + 10 ** (-dif / 400)); W = 1.0 if r.margin > 0 else (0.5 if r.margin == 0 else 0.0)
        dif_w = dif if r.margin >= 0 else -dif
        mult = np.log(abs(r.margin) + 1) * 2.2 / (0.001 * dif_w + 2.2)
        s = K * mult * (W - p); elo[r.home] = eh + s; elo[r.away] = ea - s
    g["pe_home"] = pre_h; g["pe_away"] = pre_a
    return g[["gid", "pe_home", "pe_away"]]

rows = []; keep = None
for K in (20, 15, 25):
    e = run_elo(K=K)
    x = m.merge(e, on="gid", how="inner")
    x["pure_lin"] = -((x.pe_home - x.pe_away) / 25 + x.hfa_pts)      # no QB term, nfelo's HFA
    x["pure_qb"] = x.pure_lin - x.qb_pts                               # + nfelo's QB term
    x["err_pure"] = x.margin + x.pure_lin; x["err_pure_qb"] = x.margin + x.pure_qb
    if K == 20: keep = x
    for per, d in {"train 2009-21": x[x.train], "test 2022-25": x[x.test], "all 2009-25": x}.items():
        for lab, a, b in [("pure vs close", "err_pure", "err_mkt"), ("pure vs nfelo_noqb (both no QB)", "err_pure", "err_nfelo_noqb"),
                          ("pure+QB vs close", "err_pure_qb", "err_mkt"), ("pure+QB vs nfelo_lin", "err_pure_qb", "err_nfelo_lin")]:
            dm, lo, hi, p, n = paired_mae_test(d[a].values, d[b].values)
            rows.append(dict(K=K, period=per, comparison=lab, n=n, MAE_a=d[a].abs().mean(), MAE_b=d[b].abs().mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))
x = keep
print(f"\nK=20: corr(pure_lin, nfelo_noqb) = {np.corrcoef(x.pure_lin, x.nfelo_noqb)[0,1]:.3f}; mean|pure_lin - nfelo_noqb| = {np.abs(x.pure_lin - x.nfelo_noqb).mean():.2f} pts; "
      f"mean|pure_lin - close| = {np.abs(x.pure_lin - x.mkt).mean():.2f} vs |nfelo_lin - close| = {np.abs(x.nfelo_lin - x.mkt).mean():.2f}")
# scale check for the pure elo (is 25/pt right for a results-only elo too?)
for per, d in {"train": x[x.train], "test": x[x.test], "all": x}.items():
    r = sm.OLS(d.margin.values, (-d.pure_lin).values.astype(float)).fit(cov_type="HC1")
    print(f"  {per:5s} calibration slope of pure Elo line (25/pt): {r.params[0]:.3f} ± {1.96*r.bse[0]:.3f}")
# shrinkage weight vs close for the pure engine
for per, d in {"train": x[x.train], "test": x[x.test], "all": x}.items():
    r = sm.OLS(d.err_mkt.values, (d.mkt - d.pure_qb).values.astype(float)).fit(cov_type="HC1")
    print(f"  {per:5s} OLS engine weight vs close for pure+QB: w = {r.params[0]:.3f} ± {1.96*r.bse[0]:.3f}")
# ATS vs close
def ats(d, line, th):
    dd = d[(d[line] - d.mkt).abs() >= th]; ph = dd[line] < dd.mkt; res = dd.margin + dd.mkt
    w = int(((ph & (res > 0)) | (~ph & (res < 0))).sum()); l = int(((ph & (res < 0)) | (~ph & (res > 0))).sum())
    ci = stats.binomtest(w, w + l).proportion_ci(0.95); return f"{w}-{l} ({w/(w+l):.3f}) [{ci.low:.3f},{ci.high:.3f}] p={stats.binomtest(w, w+l).pvalue:.3f}"
for per, d in {"test 2022-25": x[x.test], "all 2009-25": x}.items():
    for th in (0.5, 2.0):
        print(f"  ATS vs close {per:12s} |gap|>={th}: pure+QB {ats(d, 'pure_qb', th)} | nfelo_lin {ats(d, 'nfelo_lin', th)}")
