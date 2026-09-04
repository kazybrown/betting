"""06_engine_sd_reconstructed.py - a REAL 3-engine disagreement SD (nfelo unregressed, 538 Elo, 538 qbelo).
nfelo_scored_individual_games.csv has per-game squared errors (538_se, qbelo_se, nfelo_unregressed_se,
market_se). With error = margin + line (ORIGINATOR convention), line = -margin +/- sqrt(se); we pick the
candidate nearest the market close. The method is validated on market_se (must reproduce mkt exactly)
and on nfelo_unregressed_se (must reproduce nfelo_b from nfelo_dif_base), and qbelo against the
published qbelo_home_line_close_rounded. Then SD3 = sample SD (ddof=1) of the three engine lines,
which is what the ORIGINATOR tag rule consumes (SD <=1.2 HIGH, 1.2-2.2 MED, >2.2 LOW).
538 Elo ended after 2022 => rolling-origin: fit <=2017, test 2018-2022 (also 2022 alone).
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/06_engine_sd_reconstructed.py
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
# scored file has no game_id: rows align with nfelo_games.csv row order? check by comparing home_line_close
print("scored rows:", len(s), "| nfelo rows:", len(n))
assert len(s) <= len(n)
nn = n.iloc[: len(s)].reset_index(drop=True)
same = (nn.home_line_close.values == s.home_line_close.values) | (nn.home_line_close.isna().values & s.home_line_close.isna().values)
print("row-aligned home_line_close agreement share:", round(same.mean(), 4))
s["game_id"] = nn.game_id.values
s["gid"] = s.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
cols = ["gid", "538_se", "qbelo_se", "nfelo_se", "nfelo_unregressed_se", "market_se", "qbelo_home_line_close_rounded", "nfelo_home_line_close_rounded"]
d = m.merge(s[cols], on="gid", how="inner")
print("merged:", len(d), "| coverage of 538_se by season:", d.groupby("season")["538_se"].apply(lambda x: x.notna().mean()).round(2).to_dict())


def recon(se, margin, ref):
    r = np.sqrt(se)
    c1 = -margin + r; c2 = -margin - r
    return np.where(np.abs(c1 - ref) <= np.abs(c2 - ref), c1, c2)


d["mkt_recon"] = recon(d.market_se, d.margin, d.mkt)
print("VALIDATION market_se -> mkt: mean |diff| =", round(np.abs(d.mkt_recon - d.mkt).mean(), 4), "| exact share =", round((np.abs(d.mkt_recon - d.mkt) < 0.01).mean(), 4))
d["nb_recon"] = recon(d.nfelo_unregressed_se, d.margin, d.mkt)
print("VALIDATION nfelo_unregressed_se -> nfelo_b: mean |diff| =", round(np.abs(d.nb_recon - d.nfelo_b).mean(), 4), "| within 0.5 share =", round((np.abs(d.nb_recon - d.nfelo_b) < 0.5).mean(), 4), "| corr =", round(np.corrcoef(d.nb_recon.fillna(0), d.nfelo_b)[0, 1], 4))
d["qbelo_recon"] = recon(d.qbelo_se, d.margin, d.mkt)
ok = d.qbelo_home_line_close_rounded.notna()
print("VALIDATION qbelo_se -> qbelo_home_line_close_rounded (rounded to .5): mean |diff| =", round(np.abs(d.qbelo_recon - d.qbelo_home_line_close_rounded)[ok].mean(), 4), "| within 0.5 share =", round((np.abs(d.qbelo_recon - d.qbelo_home_line_close_rounded) <= 0.5)[ok].mean(), 4))
d["l538"] = recon(d["538_se"], d.margin, d.mkt)
# ambiguity: when both candidates are within 6 pts of the market the pick is uncertain
amb = (np.abs(2 * np.sqrt(d["538_se"])) < 6)
print("538 reconstruction ambiguous share (|err|<3 so both candidates within 6 of each other):", round(amb.mean(), 3), "-> in those games D is small either way; kept.")

e = d.dropna(subset=["l538", "qbelo_recon", "nfelo_b"]).copy()
E = e[["nfelo_b", "l538", "qbelo_recon"]].values
e["mean3"] = E.mean(axis=1)
e["sd3"] = E.std(axis=1, ddof=1)
e["D3_mkt"] = (e.mean3 - e.mkt).abs()
e["e_m3"] = e.margin + e.mean3; e["ae_m3"] = e.e_m3.abs()
print("\nengine lines pairwise corr:"); print(e[["nfelo_b", "l538", "qbelo_recon", "mkt"]].corr().round(3).to_string())
print("sd3 distribution:", e.sd3.describe().round(2).to_dict())
print("share tagged by literal thresholds on sd3: HIGH(<=1.2) =", round((e.sd3 <= 1.2).mean(), 3), "MED =", round(((e.sd3 > 1.2) & (e.sd3 <= 2.2)).mean(), 3), "LOW(>2.2) =", round((e.sd3 > 2.2).mean(), 3))
print("Spearman(sd3, D3_mkt) =", round(stats.spearmanr(e.sd3, e.D3_mkt)[0], 3), "| Spearman(sd3, D_base) =", round(stats.spearmanr(e.sd3, e.D_base)[0], 3))

fit = e[e.season <= 2017]; test = e[(e.season >= 2018) & (e.season <= 2022)]
print(f"\nrolling-origin: fit n={len(fit)} (<=2017), test n={len(test)} (2018-2022)")


def block(target, x, label):
    f, t = fit.dropna(subset=[target, x]), test.dropna(subset=[target, x])
    rf = sm.OLS(f[target].values, sm.add_constant(f[x].values)).fit(cov_type="HC1")
    rt = sm.OLS(t[target].values, sm.add_constant(t[x].values)).fit(cov_type="HC1")
    rho, p = stats.spearmanr(t[x], t[target])
    print(f"\n{label}: {target} ~ {x} | fit slope={rf.params[1]:.3f} (p={rf.pvalues[1]:.3f}) | TEST slope={rt.params[1]:.3f} se={rt.bse[1]:.3f} p={rt.pvalues[1]:.3f} | Spearman={rho:.3f} p={p:.3f}")
    return rt.params[1], rt.pvalues[1], rho, p


res = {}
for target, lab in [("ae_mkt", "market |err|"), ("ae_m3", "3-engine-mean |err|"), ("ae_nb", "nfelo-base |err|")]:
    res[(target, "sd3")] = block(target, "sd3", lab)
    res[(target, "D3_mkt")] = block(target, "D3_mkt", lab)

print("\nLiteral ORIGINATOR tag thresholds on sd3, TEST 2018-2022: realized error by tag")
tag = np.where(test.sd3 <= 1.2, "HIGH", np.where(test.sd3 <= 2.2, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    t = test[tag == k]
    W, L, P = ats(t.mean3, t.mkt, t.margin)
    print(f"  {k}: n={len(t)} share={len(t)/len(test):.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} rmse_engine_mean={np.sqrt((t.e_m3**2).mean()):.2f} mean|err| mkt={t.ae_mkt.mean():.2f} eng={t.ae_m3.mean():.2f} | mean D3_mkt={t.D3_mkt.mean():.2f} | engine-mean vs mkt ATS {W}-{L}-{P} ({W/(W+L):.3f})")
h, l = test[tag == "HIGH"], test[tag == "LOW"]
F = (l.e_mkt**2).mean() / (h.e_mkt**2).mean(); pF = 1 - stats.f.cdf(F, len(l) - 1, len(h) - 1)
F2 = (l.e_m3**2).mean() / (h.e_m3**2).mean(); pF2 = 1 - stats.f.cdf(F2, len(l) - 1, len(h) - 1)
print(f"  variance ratio LOW/HIGH: market {F:.3f} (p={pF:.3f}) | engine mean {F2:.3f} (p={pF2:.3f})")
print("Same, FIT era <=2017 (in-sample, reference):")
tagf = np.where(fit.sd3 <= 1.2, "HIGH", np.where(fit.sd3 <= 2.2, "MED", "LOW"))
for k in ["HIGH", "MED", "LOW"]:
    t = fit[tagf == k]
    print(f"  {k}: n={len(t)} share={len(t)/len(fit):.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} rmse_engine_mean={np.sqrt((t.e_m3**2).mean()):.2f} | mean D3_mkt={t.D3_mkt.mean():.2f}")

print("\nDoes sd3 add anything beyond D3_mkt for the engine-mean error? (test era, OLS |err| ~ sd3 + D3_mkt)")
X = sm.add_constant(test[["sd3", "D3_mkt"]].values); r = sm.OLS(test.ae_m3.values, X).fit(cov_type="HC1")
print(f"  sd3 coef={r.params[1]:.3f} (p={r.pvalues[1]:.3f}) | D3_mkt coef={r.params[2]:.3f} (p={r.pvalues[2]:.3f})")
print("Mechanical check on engine mean: sqrt(rmse_mkt^2 + mean(D3^2)) vs realized rmse_engine_mean by D3 tercile (test):")
lab = pd.qcut(test.D3_mkt, 3, labels=["T1", "T2", "T3"])
for k in ["T1", "T2", "T3"]:
    t = test[lab == k]
    print(f"  {k}: n={len(t)} D3 mean={t.D3_mkt.mean():.2f} rmse_mkt={np.sqrt((t.e_mkt**2).mean()):.2f} predicted={np.sqrt((t.e_mkt**2).mean() + (t.D3_mkt**2).mean()):.2f} realized={np.sqrt((t.e_m3**2).mean()):.2f}")
print("\nWho is right when engines disagree among themselves? blend weight w of engine mean vs market by sd3 tercile (fit and test)")
for era, dd in [("fit<=2017", fit), ("test2018-22", test)]:
    lab = pd.qcut(dd.sd3, 3, labels=["T1", "T2", "T3"])
    out = []
    for k in ["T1", "T2", "T3"]:
        t = dd[lab == k]; x = -(t.mean3 - t.mkt)
        r = sm.OLS(t.e_mkt.values, x.values).fit(cov_type="HC1")
        out.append(f"{k}: n={len(t)} w={r.params[0]:.3f} (se {r.bse[0]:.3f})")
    print(f"  {era}: " + " | ".join(out))
