"""critic_05 (additional findings). (A) Documents the generic season-level offset in the expert's
'league-relative' prior-8 features (prior-season reference; 2019 used for 2023) for every feature --
the artifact that inflates/deflates every rule's OOS dMAE in 2023-25 regardless of matchup signal.
(B) Pace as a DISPERSION signal for the totals confidence tag: does prior pace predict |total -
market| (fit 2009-19, check 2023-25)? (C) the same for realized plays (upper bound).
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from critic_common import *  # noqa
pd.set_option("display.width", 220)

m = load_games_table(); m = add_std_centered(m)
d = combos(m, "r8d"); d["spp_avg_s"] = (d.h_spp_neut_r8s + d.a_spp_neut_r8s) / 2
d["gplays_avg_s"] = (d.h_game_plays_r8s + d.a_game_plays_r8s) / 2
d = reg_sample(d, BASE + ["lg_blend", "spp_avg", "gplays_avg", "epa_sum"])
tr, te = d[d.train], d[d.test]

print("== (A) mean of the expert's league-relative prior-8 team feature by TEST season (should be ~0 if the reference were current) ==")
tgf = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
tgf = tgf[(tgf.game_type == "REG") & (tgf.season >= 2023)]
rows = []
for f in ["spp_neut", "plays", "game_plays", "pr_neut", "proe", "epa_off", "epa_def", "expl_off", "expl_def", "ppd", "pts"]:
    sd = tgf[f + "_r8d"].std()
    rows.append([f, sd] + [tgf[tgf.season == y][f + "_r8d"].mean() / sd for y in (2023, 2024, 2025)])
print(pd.DataFrame(rows, columns=["feature", "SD of r8d", "2023 mean (in SD)", "2024 mean (in SD)", "2025 mean (in SD)"]).round(2).to_string(index=False))
print("  -> offsets of 0.2-0.6 SD in individual seasons; any linear rule inherits a season-level shift that is not matchup information")

print("\n== (B) prior pace as a dispersion signal: |total - market| ~ pace (HC1), fit 2009-19, check 2023-25 ==")
for c in ["spp_avg_s", "gplays_avg_s", "spp_avg", "gplays_avg"]:
    f1 = ols(tr.total_err_mkt.abs(), tr[[c]]); f2 = ols(te.total_err_mkt.abs(), te[[c]])
    print(f"  {c:12s} train {f1.params[c]:+.3f} (se {f1.bse[c]:.3f}) p={f1.pvalues[c]:.3f} [per SD {f1.params[c]*tr[c].std():+.2f} pts of MAE] | test {f2.params[c]:+.3f} (se {f2.bse[c]:.3f}) p={f2.pvalues[c]:.3f}")
f1 = ols(tr.total_err_mkt.abs(), tr[["mkt_total"]]); f2 = ols(te.total_err_mkt.abs(), te[["mkt_total"]])
print(f"  (for scale) |err| ~ market total: train {f1.params['mkt_total']:+.3f} (p={f1.pvalues['mkt_total']:.3f}) | test {f2.params['mkt_total']:+.3f} (p={f2.pvalues['mkt_total']:.3f})")
lo_q, hi_q = np.percentile(pd.concat([tr.h_spp_neut_r8s, tr.a_spp_neut_r8s]), [25, 75])
for lab, x in [("train", tr), ("test", te)]:
    ff = x[(x.h_spp_neut_r8s <= lo_q) & (x.a_spp_neut_r8s <= lo_q)]; ss = x[(x.h_spp_neut_r8s >= hi_q) & (x.a_spp_neut_r8s >= hi_q)]
    lv = stats.levene(ff.total_err_mkt, ss.total_err_mkt)
    print(f"  {lab}: SD of (total - market) FF={ff.total_err_mkt.std():.2f} (n={len(ff)}) vs SS={ss.total_err_mkt.std():.2f} (n={len(ss)}); all={x.total_err_mkt.std():.2f}; Levene p={lv.pvalue:.3f}")

print("\n== (C) realized plays (not knowable pre-kickoff; upper bound) and the spread error ==")
for lab, x in [("train", tr), ("test", te)]:
    gp = x.h_plays_act + x.a_plays_act
    f = ols(x.total_err_mkt.abs(), pd.DataFrame({"gp": gp})); fs = ols(x.spread_err_mkt.abs(), x[["spp_avg_s"]])
    print(f"  {lab}: |total err| ~ realized plays {f.params['gp']:+.3f} (p={f.pvalues['gp']:.3f}) | |spread err| ~ prior spp_avg_s {fs.params['spp_avg_s']:+.3f} (p={fs.pvalues['spp_avg_s']:.3f})")
