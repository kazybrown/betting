"""critic_04_T3_who_is_right.py - CRITIC of T3 (disagreement -> who is right; REJECTED by the expert).
 (1) rolling-origin test of the w(D) = w0 + w1*D rule (fit all prior seasons, apply to season s) vs a
     constant-w blend and vs the market, 2016-2025: per-season MAE differences and the sign of w1 refit each year;
 (2) is the in-sample w1<0 driven by the near-zero-D games where w is unidentified? refit w(D) on D>=1 only;
 (3) 'the model' = the REAL nfelo pre-regression line (historic_projected_spreads, 2021-25) instead of the
     crippled nfelo_b: OOS w 2022-25, w by D band, ATS vs close;
 (4) the vs-OPEN 53.4% ATS: per season, break-even at -110, and whether it survives in the historic file's own
     open line; is it the market moving toward nfelo or nfelo (which regresses toward the market) moving toward the close?
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_04_T3_who_is_right.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, ats

pd.set_option("display.width", 220)
m = build()
rng = np.random.default_rng(3)


def w_fit(d, model, D=None, min_D=0):
    d = d[d[D].ge(min_D)] if D else d
    x = -(d[model] - d.mkt)
    if D is None:
        r = sm.OLS(d.e_mkt.values, x.values).fit(cov_type="HC1"); return r.params[0], r.bse[0]
    X = np.column_stack([x, x * d[D]]); r = sm.OLS(d.e_mkt.values, X).fit(cov_type="HC1")
    return r.params, r.bse, r.pvalues


print("(1) ROLLING-ORIGIN w(D) rule vs constant w vs market, nfelo_b, 2016-25")
rows = []
for s in range(2016, 2026):
    f = m[m.season < s]; t = m[m.season == s]
    w0, _ = w_fit(f, "nfelo_b"); (a, b1), _, pv = w_fit(f, "nfelo_b", "D_base")
    wD = np.clip(a + b1 * t.D_base, 0, 1)
    e_c = np.abs(t.margin + t.mkt + w0 * (t.nfelo_b - t.mkt)); e_d = np.abs(t.margin + t.mkt + wD * (t.nfelo_b - t.mkt)); e_m = t.ae_mkt
    rows.append(dict(season=s, n=len(t), w_const=w0, w0=a, w1=b1, p_w1=pv[1], mae_mkt=e_m.mean(), d_const=e_c.mean() - e_m.mean(), d_wD=e_d.mean() - e_m.mean(), d_wD_vs_const=e_d.mean() - e_c.mean()))
t1 = pd.DataFrame(rows).set_index("season"); print(t1.round(4).to_string())
print(f"  w1 refit each year: negative in {(t1.w1<0).sum()}/10, p<0.05 in {(t1.p_w1<0.05).sum()}/10 | w(D) rule beats constant-w in {(t1.d_wD_vs_const<0).sum()}/10 seasons, mean {t1.d_wD_vs_const.mean():+.4f} | beats market in {(t1.d_wD<0).sum()}/10, mean {t1.d_wD.mean():+.4f}")

print("\n(2) is the in-sample w1<0 identified by the near-zero-D games? refit on <=2021 with D_base >= 1 / >= 2 only")
fit = m[m.era == "fit"]; test = m[m.era == "test"]
for md in [0, 0.5, 1, 2]:
    (a, b1), (sa, sb), pv = w_fit(fit, "nfelo_b", "D_base", min_D=md); w0, s0 = w_fit(fit[fit.D_base >= md], "nfelo_b")
    print(f"  D_base >= {md}: n={int((fit.D_base>=md).sum())} const w={w0:.3f} (se {s0:.3f}) | w(D) = {a:.3f} (se {sa:.3f}) + {b1:+.3f} (se {sb:.3f}, p={pv[1]:.3f})*D")
print("  same in TEST era:")
for md in [0, 1, 2]:
    (a, b1), (sa, sb), pv = w_fit(test, "nfelo_b", "D_base", min_D=md); w0, s0 = w_fit(test[test.D_base >= md], "nfelo_b")
    print(f"  D_base >= {md}: n={int((test.D_base>=md).sum())} const w={w0:.3f} (se {s0:.3f}) | w(D) = {a:.3f} + {b1:+.3f} (se {sb:.3f}, p={pv[1]:.3f})*D")

print("\n(3) 'the model' = REAL nfelo pre-regression line (historic_projected_spreads.csv), 2021-25")
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
j = m.merge(h[["gid", "home_line_pre_regression", "home_line_open", "home_line_close"]].rename(columns={"home_line_open": "h_open", "home_line_close": "h_close", "home_line_pre_regression": "pre"}), on="gid", how="inner")
j["D_pre"] = (j.pre - j.mkt).abs(); j["e_pre"] = j.margin + j.pre
print(f"  joined n={len(j)} seasons {sorted(j.season.unique())} | corr(pre, margin) = {np.corrcoef(j.pre, j.margin)[0,1]:+.3f} | MAE pre {j.e_pre.abs().mean():.3f} vs market {j.ae_mkt.mean():.3f} vs nfelo_b {j.ae_nb.mean():.3f}")
jt = j[j.season >= 2022]
w0, s0 = w_fit(jt, "pre"); (a, b1), (sa, sb), pv = w_fit(jt, "pre", "D_pre")
print(f"  2022-25 (n={len(jt)}): constant w = {w0:+.3f} (se {s0:.3f}) | w(D) = {a:+.3f} {b1:+.3f}*D (p_w1={pv[1]:.3f})")
for lo, hi, name in [(0, 1.5, "HIGH"), (1.5, 3, "MED"), (3, 99, "LOW")]:
    s_ = jt[(jt.D_pre >= lo) & (jt.D_pre < hi)]; w, se = w_fit(s_, "pre"); W, L, P = ats(s_.pre, s_.mkt, s_.margin)
    print(f"    {name} D_pre [{lo},{hi}): n={len(s_)} share={len(s_)/len(jt):.2f} w={w:+.3f} (se {se:.3f}) | ATS vs close {W}-{L}-{P} ({W/(W+L):.3f}, p={stats.binomtest(W, W+L).pvalue:.3f}) | rmse pre {np.sqrt((s_.e_pre**2).mean()):.2f} mkt {np.sqrt((s_.e_mkt**2).mean()):.2f} identity {np.sqrt((s_.e_mkt**2).mean() + (s_.D_pre**2).mean()):.2f}")
for w in [0.1, 0.2, 0.3, 0.5]:
    e = np.abs(jt.margin + jt.mkt + w * (jt.pre - jt.mkt)); print(f"    blend w={w}: MAE diff vs market {e.mean() - jt.ae_mkt.mean():+.4f} (paired t p={stats.ttest_rel(e, jt.ae_mkt).pvalue:.3f})")

print("\n(4) model vs OPEN line: 53.4% ATS - per season, break-even, and source")
m["e_open"] = m.margin + m.mkt_open
rows = []
for s in range(2009, 2026):
    d = m[m.season == s]; W, L, P = ats(d.nfelo_b, d.mkt_open, d.margin); Wc, Lc, Pc = ats(d.nfelo_b, d.mkt, d.margin)
    rows.append(dict(season=s, n=len(d), open_w=W, open_l=L, open_rate=W / (W + L), close_rate=Wc / (Wc + Lc), mkt_move_toward_model=float(np.mean(np.sign(d.mkt - d.mkt_open) == np.sign(d.nfelo_b - d.mkt_open))), mkt_moved=float(np.mean(d.mkt != d.mkt_open))))
t4 = pd.DataFrame(rows).set_index("season"); print(t4.round(3).to_string())
W, L = t4.open_w.sum(), t4.open_l.sum()
print(f"  ALL 2009-25 vs open: {W}-{L} ({W/(W+L):.4f}, p={stats.binomtest(int(W), int(W+L)).pvalue:.4f}); break-even at -110 = 0.5238 -> {'above' if W/(W+L) > 0.5238 else 'below'} break-even; seasons > 52.4%: {(t4.open_rate>0.5238).sum()}/17")
Wt, Lt = t4.loc[2022:, "open_w"].sum(), t4.loc[2022:, "open_l"].sum(); print(f"  2022-25 vs open: {Wt}-{Lt} ({Wt/(Wt+Lt):.4f}, p={stats.binomtest(int(Wt), int(Wt+Lt)).pvalue:.4f})")
print(f"  share of games where market moved: {t4.mkt_moved.mean():.2f}; when it moved, share toward the model side: {t4.mkt_move_toward_model.mean():.3f}")
# is this specific to nfelo, or does ANY reasonable number 'beat the open'? test with qbelo (independent) and with the CLOSE itself vs the open
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
sc = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
nn = n.iloc[: len(sc)].reset_index(drop=True); sc["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
q = m.merge(sc[["gid", "qbelo_home_line_close_rounded"]].rename(columns={"qbelo_home_line_close_rounded": "qbelo"}), on="gid", how="inner").dropna(subset=["qbelo"])
W, L, P = ats(q.qbelo, q.mkt_open, q.margin); print(f"  538 qbelo vs OPEN (2009-25, n={len(q)}): {W}-{L} ({W/(W+L):.4f}, p={stats.binomtest(W, W+L).pvalue:.4f})")
W, L, P = ats(q.qbelo, q.mkt, q.margin); print(f"  538 qbelo vs CLOSE: {W}-{L} ({W/(W+L):.4f})")
W, L, P = ats(m.mkt, m.mkt_open, m.margin); print(f"  CLOSE vs OPEN (closing line as the 'model'): {W}-{L} ({W/(W+L):.4f}, p={stats.binomtest(W, W+L).pvalue:.4f}) -> how much of the open is beatable at all")
