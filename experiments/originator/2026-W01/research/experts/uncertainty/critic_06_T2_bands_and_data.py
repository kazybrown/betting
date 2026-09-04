"""critic_06_T2_bands_and_data.py - CRITIC of T2 (thresholds) + data hygiene. (v2: totals band table now uses the market TOTAL error e_tot as reference.)
 (1) the 320 games dropped by the nfelo join: which seasons/weeks? any survivorship?
 (2) nflverse close vs nfelo close outliers (|diff| > 3): what are they, do they matter?
 (3) T2 band sensitivity: the expert's 1.5/3.0 cuts vs 1.0/2.5 vs 2.0/4.0 vs terciles -> is 'LOW band excess
     positive in all 4 seasons' special or does ANY high-D band show it (identity)? F-test LOW vs HIGH for the
     model number per cut; same with the placebo model (market + noise) to show the pattern is mechanical;
 (4) the literal SD thresholds applied to a 3-'engine' SD: nfelo_b, qbelo, and market as the third -> does the
     SD sort ANY error? (closest local analog to a 3-engine SD, but note one engine IS the market);
 (5) does sd2 (engine SD) at least predict D (distance of the mean from market)? if not, the SD tag is not
     even a proxy for the D tag.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_06_T2_bands_and_data.py
"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build
from kit import load_games, load_nfelo

pd.set_option("display.width", 220)
m = build(); test = m[m.era == "test"]; fit = m[m.era == "fit"]
rng = np.random.default_rng(11)

print("(1) games dropped by the nfelo join")
g = load_games(); g = g[(g.season >= 2009) & g.mkt_spread.notna()]
n = load_nfelo()
drop = g[~g.gid.isin(m.gid)]
print(f"  dropped n={len(drop)} | by season: {drop.season.value_counts().sort_index().to_dict()}")
print(f"  in nfelo file at all? {drop.gid.isin(n.gid).mean():.3f} share | of those, nfelo_home_line_close NaN share: {n[n.gid.isin(drop.gid)].nfelo_home_line_close.isna().mean():.3f}, nfelo_dif_base NaN share: {n[n.gid.isin(drop.gid)].nfelo_dif_base.isna().mean():.3f}")
print(f"  dropped games: market RMSE {np.sqrt((drop.spread_err_mkt**2).mean()):.2f} vs kept {np.sqrt((m.e_mkt**2).mean()):.2f}; mean |spread| {drop.mkt_spread.abs().mean():.2f} vs {m.abs_mkt.mean():.2f}; weeks {drop.week.value_counts().sort_index().head(6).to_dict()}")

print("\n(2) nflverse close vs nfelo close outliers")
d = m.assign(diff=(m.mkt - m.home_line_close)); o = d[d["diff"].abs() > 3]
print(f"  |diff|>3: n={len(o)}"); print(o[["gid", "mkt", "home_line_close", "nfelo_b", "nfelo_c", "margin", "location"]].to_string(index=False))
print("  -> a handful of source discrepancies; re-run T1a slope without them: ", end="")
import statsmodels.api as sm
k = d[d["diff"].abs() <= 3]; r = sm.OLS(k.ae_mkt.values, sm.add_constant(k.D_base.values)).fit(cov_type="HC1"); print(f"pooled slope {r.params[1]:+.3f} p={r.pvalues[1]:.3f} (n={len(k)})")

print("\n(3) T2 band sensitivity (spread, model = nfelo_b, test 2022-25) - realized excess RMSE (model - market) per band and F-test LOW vs HIGH")
def band_table(df, D, e_model, cuts, label, e_ref="e_mkt"):
    lo_c, hi_c = cuts; out = []
    for lo, hi, name in [(0, lo_c, "HIGH"), (lo_c, hi_c, "MED"), (hi_c, 99, "LOW")]:
        s_ = df[(df[D] >= lo) & (df[D] < hi)]; rm, rk = np.sqrt((s_[e_model]**2).mean()), np.sqrt((s_[e_ref]**2).mean())
        out.append(dict(band=name, n=len(s_), share=len(s_)/len(df), rmse_model=rm, rmse_mkt=rk, excess=rm - rk, identity_excess=np.sqrt(rk**2 + (s_[D]**2).mean()) - rk))
    t = pd.DataFrame(out); h, l = df[df[D] < lo_c], df[df[D] >= hi_c]
    F = (l[e_model]**2).mean() / (h[e_model]**2).mean(); pF = 1 - stats.f.cdf(F, len(l)-1, len(h)-1)
    per = [np.sqrt((df[(df.season==s)&(df[D]>=hi_c)][e_model]**2).mean()) - np.sqrt((df[(df.season==s)&(df[D]>=hi_c)][e_ref]**2).mean()) for s in sorted(df.season.unique())]
    print(f"  {label} cuts {cuts}: " + " | ".join(f"{r.band} n={r.n} share={r.share:.2f} excess={r.excess:+.2f} (identity {r.identity_excess:+.2f})" for r in t.itertuples()) + f" || var ratio LOW/HIGH {F:.3f} p={pF:.3f} | LOW excess by season {np.round(per,2)}")
for cuts in [(1.5, 3.0), (1.0, 2.5), (2.0, 4.0), (1.0, 2.0)]:
    band_table(test, "D_base", "e_nb", cuts, "nfelo_b")
print("  PLACEBO model = market + noise (|d| resampled from D_base), same cuts, 300 reps: mean LOW-band excess, share of reps with LOW excess > 0 in all 4 seasons")
for cuts in [(1.5, 3.0), (1.0, 2.5)]:
    ex, all4 = [], []
    for _ in range(300):
        dd = rng.choice(test.D_base.values, len(test)) * rng.choice([-1, 1], len(test)); e_f = test.e_mkt.values + dd; Df = np.abs(dd)
        lo = Df >= cuts[1]; ex.append(np.sqrt((e_f[lo]**2).mean()) - np.sqrt((test.e_mkt.values[lo]**2).mean()))
        per = [np.sqrt((e_f[lo & (test.season.values==s)]**2).mean()) - np.sqrt((test.e_mkt.values[lo & (test.season.values==s)]**2).mean()) for s in [2022, 2023, 2024, 2025]]
        all4.append(all(p > 0 for p in per))
    print(f"    cuts {cuts}: placebo LOW excess mean {np.mean(ex):+.2f} [{np.quantile(ex,.025):+.2f},{np.quantile(ex,.975):+.2f}] | 'positive in all 4 seasons' in {100*np.mean(all4):.0f}% of placebo reps | real nfelo_b: see above")
print("  TOTALS (model = T_elo):")
for cuts in [(2.5, 5.0), (2.0, 4.0), (3.0, 6.0)]:
    band_table(test, "D_tot", "e_telo", cuts, "T_elo", e_ref="e_tot")

print("\n(4) a 3-number SD (nfelo_b, qbelo, market close) with the literal 1.2/2.2 cuts -> does it sort error?")
nf = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
sc = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
nn = nf.iloc[: len(sc)].reset_index(drop=True); sc["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
q = m.merge(sc[["gid", "qbelo_home_line_close_rounded"]].rename(columns={"qbelo_home_line_close_rounded": "qbelo"}), on="gid", how="inner").dropna(subset=["qbelo"]).copy()
q["sd3"] = q[["nfelo_b", "qbelo", "mkt"]].std(axis=1, ddof=1); q["mean3"] = q[["nfelo_b", "qbelo", "mkt"]].mean(axis=1); q["e_m3"] = q.margin + q.mean3; q["D3"] = (q.mean3 - q.mkt).abs()
q["sd2"] = (q.nfelo_b - q.qbelo).abs() / np.sqrt(2); q["mean2"] = (q.nfelo_b + q.qbelo) / 2; q["e_m2"] = q.margin + q.mean2; q["D2"] = (q.mean2 - q.mkt).abs()
qt = q[q.era == "test"]
print(f"  sd3 shares: HIGH<=1.2 {(qt.sd3<=1.2).mean():.2f} MED {((qt.sd3>1.2)&(qt.sd3<=2.2)).mean():.2f} LOW>2.2 {(qt.sd3>2.2).mean():.2f}")
for k_, lo, hi in [("HIGH", 0, 1.2), ("MED", 1.2, 2.2), ("LOW", 2.2, 99)]:
    s_ = qt[(qt.sd3 > lo) & (qt.sd3 <= hi)] if k_ != "HIGH" else qt[qt.sd3 <= hi]
    print(f"    {k_}: n={len(s_)} rmse_mkt {np.sqrt((s_.e_mkt**2).mean()):.2f} rmse_mean3 {np.sqrt((s_.e_m3**2).mean()):.2f} rmse_mean2 {np.sqrt((s_.e_m2**2).mean()):.2f} | mean D3 {s_.D3.mean():.2f}")
print(f"  Spearman(sd3, |e_mkt|) test = {stats.spearmanr(qt.sd3, qt.ae_mkt)[0]:+.3f} (p={stats.spearmanr(qt.sd3, qt.ae_mkt)[1]:.3f}) | Spearman(sd3, |e_mean3|) = {stats.spearmanr(qt.sd3, qt.e_m3.abs())[0]:+.3f} (p={stats.spearmanr(qt.sd3, qt.e_m3.abs())[1]:.3f}) | note sd3 contains the market so it is partly D by construction: Spearman(sd3, D3) = {stats.spearmanr(qt.sd3, qt.D3)[0]:+.3f}")

print("\n(5) is the engine SD even a proxy for distance-to-market? (pooled 2009-25)")
print(f"  Spearman(sd2, D2) = {stats.spearmanr(q.sd2, q.D2)[0]:+.3f} | Spearman(sd2, D_base) = {stats.spearmanr(q.sd2, q.D_base)[0]:+.3f} | Spearman(sd2, |e_m2|) = {stats.spearmanr(q.sd2, q.e_m2.abs())[0]:+.3f} (p={stats.spearmanr(q.sd2, q.e_m2.abs())[1]:.3f}) | Spearman(D2, |e_m2|) = {stats.spearmanr(q.D2, q.e_m2.abs())[0]:+.3f} (p={stats.spearmanr(q.D2, q.e_m2.abs())[1]:.3f})")
r = sm.OLS(q.e_m2.abs().values, sm.add_constant(q[["sd2", "D2"]].values)).fit(cov_type="HC1"); print(f"  pooled |e_mean2| ~ sd2 + D2: sd2 {r.params[1]:+.3f} (p={r.pvalues[1]:.3f}) D2 {r.params[2]:+.3f} (p={r.pvalues[2]:.3f}) n={len(q)}")
