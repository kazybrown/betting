"""CRITIC 02 (T2: market-blind loss and shrinkage). Alternative specifications:
A. loss vs close by era (paired), median |err| and RMSE, REG only;
B. ATS by era and by test season (binomial CIs) -- is the 2009-25 52.5% driven by early seasons?
C. shrinkage w by era (OLS, HC1) and a recency-window rolling w (last 4 seasons only);
D. LEAKAGE CHECK: are nfelo's Elo UPDATES market-informed? Regress each team's Elo change (next game's starting Elo
   minus this game's) on its own-forecast error vs the market error. If the market error carries the weight, the
   'market-blind' nfelo number is not market-blind and the 0.24 MAE loss is a LOWER bound on a truly blind number.
E. Informational: the OPEN line. MAE of open vs close vs nfelo_lin; ATS of nfelo_lin against the OPEN.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import load, boot_ci, paired_mae_test, ROOT
import sys; sys.path.insert(0, str(ROOT))
from kit import load_nfelo, load_games
pd.set_option("display.width", 220)
m = load(verbose=False)
ERAS = {"2009-13": (2009, 2013), "2014-18": (2014, 2018), "2019-21": (2019, 2021), "2022-25": (2022, 2025)}

print("=== A. nfelo_lin loss vs close by era (paired), plus median |err| and RMSE ===")
rows = []
for era, (a, b) in ERAS.items():
    d = m[(m.season >= a) & (m.season <= b)]
    dm, lo, hi, p, n = paired_mae_test(d.err_nfelo_lin.values, d.err_mkt.values)
    dr = d[~d.post]; dm2, lo2, hi2, p2, n2 = paired_mae_test(dr.err_nfelo_lin.values, dr.err_mkt.values)
    rows.append(dict(era=era, n=n, MAE_nfelo=d.err_nfelo_lin.abs().mean(), MAE_mkt=d.err_mkt.abs().mean(), dMAE=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p,
                     medAE_nfelo=d.err_nfelo_lin.abs().median(), medAE_mkt=d.err_mkt.abs().median(), RMSE_nfelo=np.sqrt((d.err_nfelo_lin**2).mean()), RMSE_mkt=np.sqrt((d.err_mkt**2).mean()),
                     dMAE_REG=dm2, p_REG=p2, gap=np.abs(d.nfelo_lin - d.mkt).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. ATS of nfelo_lin side vs close, by era and by test season (push excluded) ===")
def ats(d, th):
    dd = d[(d.nfelo_lin - d.mkt).abs() >= th]
    ph = dd.nfelo_lin < dd.mkt; res = dd.margin + dd.mkt
    w = int(((ph & (res > 0)) | (~ph & (res < 0))).sum()); l = int(((ph & (res < 0)) | (~ph & (res > 0))).sum())
    ci = stats.binomtest(w, w + l).proportion_ci(0.95)
    return w, l, w / (w + l), f"[{ci.low:.3f},{ci.high:.3f}]", stats.binomtest(w, w + l).pvalue
rows = []
for era, (a, b) in list(ERAS.items()) + [(f"{s}", (s, s)) for s in (2022, 2023, 2024, 2025)]:
    d = m[(m.season >= a) & (m.season <= b)]
    for th in (0.5, 2.0):
        w, l, pct, ci, p = ats(d, th); rows.append(dict(period=era, thresh=th, W=w, L=l, ats=pct, ci=ci, p_vs_50=p))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== C. Shrinkage weight w (OLS err_mkt ~ (mkt - nfelo_lin), HC1) by era; recency-window rolling w (fit on the previous 4 seasons only) ===")
for era, (a, b) in ERAS.items():
    d = m[(m.season >= a) & (m.season <= b)]
    r = sm.OLS(d.err_mkt.values, (d.mkt - d.nfelo_lin).values.astype(float)).fit(cov_type="HC1")
    print(f"  {era}: w = {r.params[0]:.3f} ± {1.96*r.bse[0]:.3f} (n={int(r.nobs)})")
grid = np.arange(0, 1.0001, 0.02); pool = []; picks = []
for t_ in range(2013, 2026):
    trn = m[(m.season < t_) & (m.season >= t_ - 4)]; tst = m[m.season == t_]
    w = grid[int(np.argmin([np.abs(trn.margin + w * trn.nfelo_lin + (1 - w) * trn.mkt).mean() for w in grid]))]; picks.append((t_, round(float(w), 2)))
    pool.append(pd.DataFrame({"e": tst.margin + w * tst.nfelo_lin + (1 - w) * tst.mkt, "em": tst.err_mkt}))
pool = pd.concat(pool); dm, lo, hi, p, n = paired_mae_test(pool.e, pool.em)
print("  4-season-window w picks:", picks)
print(f"  rolling(4y) blend vs market: MAE {np.abs(pool.e).mean():.3f} vs {np.abs(pool.em).mean():.3f} dMAE {dm:+.3f} CI[{lo:.3f},{hi:.3f}] p={p:.3f} n={n}")

print("\n=== D. LEAKAGE CHECK: are nfelo Elo updates driven by the market error rather than nfelo's own forecast error? ===")
n = load_nfelo(); n = n[n.season <= 2025]
g = load_games()[["gid", "margin", "mkt_spread", "game_type"]]
x = n.merge(g, on="gid", how="inner")
x["nfelo_lin"] = -x.nfelo_dif_base / 25.0
long = []
for side, sgn in (("home", 1), ("away", -1)):
    long.append(pd.DataFrame({"team": x[side], "season": x.season, "week": x.week, "elo": x[f"starting_nfelo_{side}"],
                              "e_own": sgn * (x.margin + x.nfelo_lin), "e_close": sgn * (x.margin + x.nfelo_home_line_close), "e_mkt": sgn * (x.margin + x.mkt_spread),
                              "mg": sgn * x.margin, "post": x.game_type != "REG"}))
L = pd.concat(long).sort_values(["team", "season", "week"])
L["elo_next"] = L.groupby(["team", "season"]).elo.shift(-1)
L["d_elo"] = L.elo_next - L.elo
L = L[L.d_elo.notna() & L.e_mkt.notna() & L.e_close.notna() & ~L.post]
print(f"  team-games with a within-season next game: n={len(L)}; corr(e_own, e_mkt)={np.corrcoef(L.e_own, L.e_mkt)[0,1]:.3f}")
for lab, cols in [("d_elo ~ e_own", ["e_own"]), ("d_elo ~ e_mkt", ["e_mkt"]), ("d_elo ~ e_close (nfelo regressed line)", ["e_close"]),
                  ("d_elo ~ e_own + e_mkt", ["e_own", "e_mkt"]), ("d_elo ~ e_own + e_close", ["e_own", "e_close"]), ("d_elo ~ e_own + e_mkt + e_close", ["e_own", "e_mkt", "e_close"])]:
    r = sm.OLS(L.d_elo.values, sm.add_constant(L[cols].values.astype(float))).fit(cov_type="HC1")
    print(f"  {lab:40s} R2={r.rsquared:.3f} " + " ".join(f"{c}={r.params[i+1]:+.3f}±{1.96*r.bse[i+1]:.3f}" for i, c in enumerate(cols)))
# non-parametric: within bins of e_own, does d_elo still vary with (e_mkt - e_own) = (mkt - own line)?
L["diff_line"] = L.e_mkt - L.e_own
r = sm.OLS(L.d_elo.values, sm.add_constant(np.column_stack([L.e_own.values, L.diff_line.values]))).fit(cov_type="HC1")
print(f"  d_elo ~ e_own + (mkt_line - own_line): coef on line difference = {r.params[2]:+.3f} ± {1.96*r.bse[2]:.3f}  (0 => update ignores the market; = coef on e_own => update uses the market line instead of its own)")
print(f"  coef on e_own in that model = {r.params[1]:+.3f}; ratio (market share of the update) = {r.params[2]/r.params[1]:.2f}")
for era, (a, b) in ERAS.items():
    d = L[(L.season >= a) & (L.season <= b)]
    r = sm.OLS(d.d_elo.values, sm.add_constant(np.column_stack([d.e_own.values, d.diff_line.values]))).fit(cov_type="HC1")
    print(f"    {era}: e_own {r.params[1]:+.3f}±{1.96*r.bse[1]:.3f}  line-diff {r.params[2]:+.3f}±{1.96*r.bse[2]:.3f}  share {r.params[2]/r.params[1]:.2f}  n={int(r.nobs)}")

print("\n=== E. Informational: OPEN line (nfelo_games.home_line_open; same sign convention, same caveat on a few sign errors) ===")
o = m[m.home_line_open.notna()].copy(); o["err_open"] = o.margin + o.home_line_open
for per, d in {"train 2009-21": o[o.train], "test 2022-25": o[o.test], "all": o}.items():
    dm, lo, hi, p, n_ = paired_mae_test(d.err_open.values, d.err_mkt.values)
    dm2, lo2, hi2, p2, _ = paired_mae_test(d.err_nfelo_lin.values, d.err_open.values)
    print(f"  {per:14s} n={n_} MAE open {d.err_open.abs().mean():.3f} vs close {d.err_mkt.abs().mean():.3f} (dMAE {dm:+.3f} [{lo:.3f},{hi:.3f}] p={p:.3f}) | nfelo_lin vs open dMAE {dm2:+.3f} [{lo2:.3f},{hi2:.3f}] p={p2:.3f} | mean|open-close| {np.abs(d.home_line_open-d.mkt).mean():.2f}")
def ats_open(d, th):
    dd = d[(d.nfelo_lin - d.home_line_open).abs() >= th]
    ph = dd.nfelo_lin < dd.home_line_open; res = dd.margin + dd.home_line_open
    w = int(((ph & (res > 0)) | (~ph & (res < 0))).sum()); l = int(((ph & (res < 0)) | (~ph & (res > 0))).sum())
    ci = stats.binomtest(w, w + l).proportion_ci(0.95); return w, l, w / (w + l), f"[{ci.low:.3f},{ci.high:.3f}]", stats.binomtest(w, w + l).pvalue
for per, d in {"test 2022-25": o[o.test], "all": o}.items():
    for th in (0.5, 1.0, 2.0, 3.0):
        w, l, pct, ci, p = ats_open(d, th); print(f"  ATS nfelo_lin vs OPEN {per:12s} |gap|>={th}: {w}-{l} ({pct:.3f}) {ci} p={p:.3f}")
# does the engine predict line movement open->close? (informational)
for per, d in {"test 2022-25": o[o.test], "all": o}.items():
    r = sm.OLS((d.mkt - d.home_line_open).values, sm.add_constant((d.nfelo_lin - d.home_line_open).values.astype(float))).fit(cov_type="HC1")
    print(f"  {per}: (close - open) ~ (nfelo_lin - open): slope {r.params[1]:.3f} ± {1.96*r.bse[1]:.3f} R2={r.rsquared:.3f}")
