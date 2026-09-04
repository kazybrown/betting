"""08_engine_sd_pure_2engine.py - engine-vs-engine SD WITHOUT reconstruction bias.
06 reconstructed 538/qbelo lines from squared errors by picking the candidate nearest the market;
that pick is wrong precisely when the engine's error is small and same-signed as its disagreement,
and the wrong candidate is CLOSER to the market -> reconstructed D shrinks when |err| is small ->
spurious positive D->|err| correlation. Here only lines that exist in the data are used:
  nfelo_b  (unregressed nfelo, from nfelo_dif_base)  and  qbelo_pub (538 QB-adjusted Elo line,
  qbelo_home_line_close_rounded, published; 538 Elo never used market lines).
  sd2 = |nfelo_b - qbelo_pub| / sqrt(2)  (sample SD of two numbers), mean2, D2_mkt = |mean2 - mkt|.
Tests: |err| ~ sd2 and ~ D2_mkt for market / engine-mean / nfelo; literal 1.2/2.2 thresholds on sd2;
blend weight by sd2 tercile. Split: fit <=2021, test 2022-25 if qbelo exists there, else rolling.
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/08_engine_sd_pure_2engine.py
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build, ats

pd.set_option("display.width", 220)
m = build()
s = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
nn = n.iloc[: len(s)].reset_index(drop=True)
assert ((nn.home_line_close.values == s.home_line_close.values) | (nn.home_line_close.isna().values & s.home_line_close.isna().values)).all()
s["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
d = m.merge(s[["gid", "qbelo_home_line_close_rounded", "qbelo_se", "538_se"]], on="gid", how="inner")
d = d.rename(columns={"qbelo_home_line_close_rounded": "qbelo_pub"})
print("qbelo_pub coverage by season:", d.groupby("season").qbelo_pub.apply(lambda x: x.notna().mean()).round(2).to_dict())
d = d.dropna(subset=["qbelo_pub"]).copy()
print("sign check corr(qbelo_pub, margin) =", round(np.corrcoef(d.qbelo_pub, d.margin)[0, 1], 3), "(strongly negative)")
# consistency: qbelo_se should equal (margin + qbelo_pub)^2 up to rounding
chk = np.abs(np.sqrt(d.qbelo_se) - np.abs(d.margin + d.qbelo_pub))
print("qbelo_se vs (margin+qbelo_pub)^2: mean |sqrt diff| =", round(chk.mean(), 3), "| within 0.5 share =", round((chk <= 0.5).mean(), 3))
d["e_q"] = d.margin + d.qbelo_pub
d["mean2"] = (d.nfelo_b + d.qbelo_pub) / 2
d["sd2"] = (d.nfelo_b - d.qbelo_pub).abs() / np.sqrt(2)
d["D2_mkt"] = (d.mean2 - d.mkt).abs()
d["e_m2"] = d.margin + d.mean2; d["ae_m2"] = d.e_m2.abs(); d["ae_q"] = d.e_q.abs()
print("corr nfelo_b vs qbelo_pub:", round(np.corrcoef(d.nfelo_b, d.qbelo_pub)[0, 1], 3), "| sd2 distribution:", d.sd2.describe().round(2).to_dict())
print("literal thresholds on sd2: HIGH(<=1.2) share =", round((d.sd2 <= 1.2).mean(), 3), "MED =", round(((d.sd2 > 1.2) & (d.sd2 <= 2.2)).mean(), 3), "LOW(>2.2) =", round((d.sd2 > 2.2).mean(), 3))
print("Spearman(sd2, D2_mkt) =", round(stats.spearmanr(d.sd2, d.D2_mkt)[0], 3), "| Spearman(sd2, D_base) =", round(stats.spearmanr(d.sd2, d.D_base)[0], 3))
fit, test = d[d.season <= 2021], d[d.season >= 2022]
print(f"fit n={len(fit)} (<=2021) | test n={len(test)} (2022-25)")
print("baseline test MAE: market", round(test.ae_mkt.mean(), 3), "| nfelo_b", round(test.ae_nb.mean(), 3), "| qbelo", round(test.ae_q.mean(), 3), "| engine mean2", round(test.ae_m2.mean(), 3))


def block(target, x, label):
    f, t = fit, test
    rf = sm.OLS(f[target].values, sm.add_constant(f[x].values)).fit(cov_type="HC1")
    rt = sm.OLS(t[target].values, sm.add_constant(t[x].values)).fit(cov_type="HC1")
    rho, p = stats.spearmanr(t[x], t[target])
    print(f"{label:22s}: {target:6s} ~ {x:6s} | fit slope={rf.params[1]:+.3f} (p={rf.pvalues[1]:.3f}) | TEST slope={rt.params[1]:+.3f} se={rt.bse[1]:.3f} p={rt.pvalues[1]:.3f} | Spearman={rho:+.3f} p={p:.3f}")


print("\n--- does engine-vs-engine SD (sd2) predict |err|? ---")
for target, lab in [("ae_mkt", "market |err|"), ("ae_m2", "engine-mean |err|"), ("ae_nb", "nfelo-base |err|"), ("ae_q", "qbelo |err|")]:
    block(target, "sd2", lab)
print("--- does engine-mean vs market distance (D2_mkt) predict |err|? (no reconstruction involved) ---")
for target, lab in [("ae_mkt", "market |err|"), ("ae_m2", "engine-mean |err|"), ("ae_nb", "nfelo-base |err|"), ("ae_q", "qbelo |err|")]:
    block(target, "D2_mkt", lab)
print("--- joint (test): ae_m2 ~ sd2 + D2_mkt ---")
X = sm.add_constant(test[["sd2", "D2_mkt"]].values); r = sm.OLS(test.ae_m2.values, X).fit(cov_type="HC1")
print(f"  sd2 coef={r.params[1]:+.3f} (p={r.pvalues[1]:.3f}) | D2_mkt coef={r.params[2]:+.3f} (p={r.pvalues[2]:.3f})")
X = sm.add_constant(test[["sd2", "D2_mkt"]].values); r = sm.OLS(test.ae_mkt.values, X).fit(cov_type="HC1")
print(f"  (market |err|) sd2 coef={r.params[1]:+.3f} (p={r.pvalues[1]:.3f}) | D2_mkt coef={r.params[2]:+.3f} (p={r.pvalues[2]:.3f})")

print("\n--- literal ORIGINATOR thresholds on sd2 (HIGH<=1.2, MED 1.2-2.2, LOW>2.2), TEST 2022-25 ---")
tag = np.where(test.sd2 <= 1.2, "HIGH", np.where(test.sd2 <= 2.2, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    t = test[tag == k]; W, L, P = ats(t.mean2, t.mkt, t.margin)
    print(f"  {k}: n={len(t)} share={len(t)/len(test):.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} rmse_engine_mean={np.sqrt((t.e_m2**2).mean()):.2f} rmse_nfelo_b={np.sqrt((t.e_nb**2).mean()):.2f} | mean|err| mkt={t.ae_mkt.mean():.2f} eng={t.ae_m2.mean():.2f} | mean D2_mkt={t.D2_mkt.mean():.2f} | engine-mean vs mkt ATS {W}-{L}-{P} ({W/(W+L):.3f}, p={stats.binomtest(W, W+L).pvalue:.3f})")
h, l = test[tag == "HIGH"], test[tag == "LOW"]
for col, lab in [("e_mkt", "market"), ("e_m2", "engine mean")]:
    F = (l[col]**2).mean() / (h[col]**2).mean(); pF = 1 - stats.f.cdf(F, len(l) - 1, len(h) - 1)
    print(f"  variance ratio LOW/HIGH {lab}: {F:.3f} (one-sided p={pF:.3f})")
print("  FIT era (in-sample reference):")
tagf = np.where(fit.sd2 <= 1.2, "HIGH", np.where(fit.sd2 <= 2.2, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    t = fit[tagf == k]; W, L, P = ats(t.mean2, t.mkt, t.margin)
    print(f"  {k}: n={len(t)} share={len(t)/len(fit):.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} rmse_engine_mean={np.sqrt((t.e_m2**2).mean()):.2f} | engine-mean vs mkt ATS {W}-{L}-{P} ({W/(W+L):.3f})")

print("\n--- mechanical check on the engine mean: sqrt(rmse_mkt^2 + mean(D2^2)) vs realized, by D2_mkt tercile (test) ---")
lab = pd.qcut(test.D2_mkt, 3, labels=["T1", "T2", "T3"])
for k in ["T1", "T2", "T3"]:
    t = test[lab == k]
    print(f"  {k}: n={len(t)} D2 mean={t.D2_mkt.mean():.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} predicted={np.sqrt((t.e_mkt**2).mean() + (t.D2_mkt**2).mean()):.2f} realized={np.sqrt((t.e_m2**2).mean()):.2f} | corr(e_mkt, mean2-mkt)={np.corrcoef(t.e_mkt, t.mean2-t.mkt)[0,1]:+.3f}")

print("\n--- who is right when the two engines disagree? blend weight w (engine mean vs market) by sd2 tercile ---")
for era, dd in [("fit<=2021", fit), ("test2022-25", test)]:
    lab = pd.qcut(dd.sd2, 3, labels=["T1", "T2", "T3"]); out = []
    for k in ["T1", "T2", "T3"]:
        t = dd[lab == k]; x = -(t.mean2 - t.mkt); r = sm.OLS(t.e_mkt.values, x.values).fit(cov_type="HC1")
        out.append(f"{k}(sd2<={dd.loc[lab==k,'sd2'].max():.2f}): n={len(t)} w={r.params[0]:+.3f} (se {r.bse[0]:.3f})")
    x = -(dd.mean2 - dd.mkt); r = sm.OLS(dd.e_mkt.values, x.values).fit(cov_type="HC1")
    print(f"  {era}: overall w={r.params[0]:+.3f} (se {r.bse[0]:.3f}) | " + " | ".join(out))
print("--- does the SPREAD between engines tell you which engine to trust? |err| of nfelo vs qbelo when they disagree by > 2 (test) ---")
t = test[test.sd2 > 2 / np.sqrt(2)]
print(f"  n={len(t)}: MAE nfelo_b={t.ae_nb.mean():.3f} qbelo={t.ae_q.mean():.3f} mean2={t.ae_m2.mean():.3f} market={t.ae_mkt.mean():.3f}")
