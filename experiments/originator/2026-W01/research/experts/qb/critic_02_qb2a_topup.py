"""CRITIC 02 (theory qb-2a): does nfelo's 538 QB adj fully price in-season backup starts (top-up = 0)?
(A) residual after the 538 adj by season and by stint, in-season, with n — is the 2022-25 anomaly a trend or noise?
(B) rolling-origin OOS (fit prior seasons, test season t, 2016-2025): 538-adj line + fitted top-up / + fixed 0.5 / + fixed 1.0
    (any stint and stint-1 only), and the 538 adj rescaled by the fitted slope; same on nfelo's regressed close.
(C) the fit-window pooled all-stint residual (+0.79 p=0.05 full sample): what is it on <=2021 only?
Reads qb_games_defs.csv. Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import mae
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna() & m.nfelo_home_line_close.notna()].copy()
m["late"] = ((m.game_type != "REG") | (m.week >= 17)).astype(int)
m["resid_close"] = m.margin + m.nfelo_home_line_close
def ns(df, lo, hi): return ((df.home_down == 1) & df.home_stint3.between(lo, hi)).astype(int) - ((df.away_down == 1) & df.away_stint3.between(lo, hi)).astype(int)
m["s1"] = ns(m, 1, 1); m["s23"] = ns(m, 2, 3); m["s4"] = ns(m, 4, 99)
ins = m[m.late == 0].copy()
def coef(y, x, d):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type="HC1"); return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())
print("sign check corr(mkt_spread, margin) =", round(np.corrcoef(m.mkt_spread, m.margin)[0, 1], 3))
print("(A) residual AFTER nfelo's 538 adj (penalty-signed, + = backup team still underperforms), in-season, any D3 stint and stint 1, by season")
for yr in range(2009, 2026):
    d = ins[ins.season == yr]
    b, se, p, n = coef("resid_base", "net_D3", d); b1, se1, p1, n1 = coef("resid_base", "s1", d)
    bm, _, _, _ = coef("resid_mkt", "net_D3", d)
    print(f"  {yr}: any-stint after538={-b:+.2f}±{se:.2f} (n_ev={n:3d})  stint-1 after538={-b1:+.2f}±{se1:.2f} (n_ev={n1:2d})   any-stint vs MARKET={-bm:+.2f}")
for lab, d in [("FIT 2009-2021", ins[ins.season <= 2021]), ("TEST 2022-2025", ins[ins.season >= 2022]), ("ALL", ins)]:
    b, se, p, n = coef("resid_base", "net_D3", d); b1, se1, p1, n1 = coef("resid_base", "s1", d); bm, sem, pm, _ = coef("resid_mkt", "net_D3", d)
    print(f"  {lab:<15s} any-stint after538={-b:+.2f}±{se:.2f} p={p:.3f} (n_ev={n})  stint-1 after538={-b1:+.2f}±{se1:.2f} p={p1:.3f} (n_ev={n1})  any-stint vs MARKET={-bm:+.2f}±{sem:.2f} p={pm:.3f}")
    r = sm.OLS(d.resid_noqb, sm.add_constant(d[["qb_adj_pts"]])).fit(cov_type="HC1")
    print(f"                  538 scale slope (1 = correctly scaled): {r.params.qb_adj_pts:.3f}±{r.bse.qb_adj_pts:.3f}")

print("\n(B) rolling-origin OOS 2016-2025, in-season games; dMAE vs the base line (negative = top-up helps); paired per season")
res = {k: [] for k in ["b538", "fit_any", "fit_s1", "fix05_any", "fix10_any", "fix10_s1", "resc", "bclose", "close_fix10_any", "close_fix10_s1", "bmkt", "mkt_fix10_s1"]}
for yr in range(2016, 2026):
    tr = ins[ins.season < yr]; te = ins[ins.season == yr]
    k_any = -sm.OLS(tr.resid_base, sm.add_constant(tr[["net_D3"]])).fit().params["net_D3"]
    k_s1 = -sm.OLS(tr.resid_base, sm.add_constant(tr[["s1"]])).fit().params["s1"]
    sc = sm.OLS(tr.resid_noqb, sm.add_constant(tr[["qb_adj_pts"]])).fit().params["qb_adj_pts"]
    e0 = te.margin + te.line_nfelo_base; ec = te.margin + te.nfelo_home_line_close; em = te.margin + te.mkt_spread
    cand = {"b538": e0, "fit_any": e0 + k_any * te.net_D3, "fit_s1": e0 + k_s1 * te.s1, "fix05_any": e0 + 0.5 * te.net_D3, "fix10_any": e0 + 1.0 * te.net_D3,
            "fix10_s1": e0 + 1.0 * te.s1, "resc": te.margin + te.line_nfelo_noqb - sc * te.qb_adj_pts,
            "bclose": ec, "close_fix10_any": ec + 1.0 * te.net_D3, "close_fix10_s1": ec + 1.0 * te.s1, "bmkt": em, "mkt_fix10_s1": em + 1.0 * te.s1}
    for k, e in cand.items(): res[k].append(np.abs(e).mean())
    print(f"  {yr}: fitted top-up any={k_any:+.2f} s1={k_s1:+.2f} scale={sc:.2f} | 538 base={res['b538'][-1]:.4f} +fit_any={res['fit_any'][-1]-res['b538'][-1]:+.4f} +fit_s1={res['fit_s1'][-1]-res['b538'][-1]:+.4f} +0.5any={res['fix05_any'][-1]-res['b538'][-1]:+.4f} +1.0any={res['fix10_any'][-1]-res['b538'][-1]:+.4f} +1.0s1={res['fix10_s1'][-1]-res['b538'][-1]:+.4f} rescaled={res['resc'][-1]-res['b538'][-1]:+.4f} | close base={res['bclose'][-1]:.4f} +1.0any={res['close_fix10_any'][-1]-res['bclose'][-1]:+.4f} +1.0s1={res['close_fix10_s1'][-1]-res['bclose'][-1]:+.4f} | mkt +1.0s1={res['mkt_fix10_s1'][-1]-res['bmkt'][-1]:+.4f}")
R = {k: np.array(v) for k, v in res.items()}
def summ(k, b):
    d = R[k] - R[b]; return f"mean dMAE={d.mean():+.4f} (seasons helped {int((d<0).sum())}/10, se over seasons {d.std(ddof=1)/np.sqrt(10):.4f})"
print("  SUMMARY 2016-25 mean over seasons:")
for k, b, lab in [("fit_any", "b538", "538 + fitted any-stint top-up"), ("fit_s1", "b538", "538 + fitted stint-1 top-up"), ("fix05_any", "b538", "538 + fixed 0.5 any-stint"), ("fix10_any", "b538", "538 + fixed 1.0 any-stint"),
                  ("fix10_s1", "b538", "538 + fixed 1.0 stint-1"), ("resc", "b538", "538 adj rescaled by fitted slope"), ("close_fix10_any", "bclose", "nfelo CLOSE + 1.0 any-stint"), ("close_fix10_s1", "bclose", "nfelo CLOSE + 1.0 stint-1"), ("mkt_fix10_s1", "bmkt", "MARKET + 1.0 stint-1")]:
    print(f"    {lab:<36s} {summ(k, b)}")
print("\n(C) fit-window (<=2021) pooled residual after 538 by stint, in-season, all games (expert's 07 sample incl. opponent-backup games):")
tr = ins[ins.season <= 2021]
r = sm.OLS(tr.resid_base, sm.add_constant(tr[["s1", "s23", "s4"]])).fit(cov_type="HC1")
print(f"  1st={-r.params.s1:+.2f}±{r.bse.s1:.2f} (p={r.pvalues.s1:.2f})  2nd-3rd={-r.params.s23:+.2f}±{r.bse.s23:.2f} (p={r.pvalues.s23:.2f})  4th+={-r.params.s4:+.2f}±{r.bse.s4:.2f} (p={r.pvalues.s4:.2f})")
