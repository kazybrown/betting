"""06: why did the pace effect vanish in 2023-25? Dispersion and persistence of prior pace, the
implied FF / SS adjustment from the 2009-2019 market-free fit, and the market's own pace weighting.
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from pathlib import Path
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "_game_features.csv", low_memory=False)
m["dome"] = m.is_dome.astype(int)
d = m[(m.game_type == "REG") & m.mkt_total.notna()].copy()
d["spp_avg"] = (d.h_spp_neut_r8d + d.a_spp_neut_r8d) / 2
d["gplays_act"] = d.h_plays_act + d.a_plays_act
d = d.dropna(subset=["spp_avg", "lg_blend", "elo_sum", "pf_sum", "pa_sum", "qb_sum", "wind_f", "div"])
tr, te = d[d.train], d[d.test]
print("prior-8 neutral seconds/play (two-team avg, league-relative): SD train=%.2f test=%.2f | realized game plays SD train=%.1f test=%.1f mean %.1f / %.1f"
      % (tr.spp_avg.std(), te.spp_avg.std(), tr.gplays_act.std(), te.gplays_act.std(), tr.gplays_act.mean(), te.gplays_act.mean()))
for lab, x in [("train 2009-19", tr), ("test 2023-25", te)]:
    r1 = stats.pearsonr(x.spp_avg, x.gplays_act); r2 = stats.pearsonr(x.gplays_act, x.total_pts); r3 = stats.pearsonr(x.spp_avg, x.mkt_total); r4 = stats.pearsonr(x.spp_avg, x.total_pts)
    b1 = np.polyfit(x.spp_avg, x.gplays_act, 1)[0]; b2 = np.polyfit(x.gplays_act, x.total_pts, 1)[0]
    print(f"  {lab}: corr(prior spp, realized plays)={r1[0]:+.3f} (slope {b1:+.2f} plays per s) | corr(realized plays, total)={r2[0]:+.3f} (slope {b2:+.3f} pts/play) "
          f"| chain = {b1*b2:+.2f} pts per s | corr(prior spp, MARKET total)={r3[0]:+.3f} | corr(prior spp, total)={r4[0]:+.3f} n={len(x)}")
# market's own pace slope (market total on prior spp, controlling for pf/pa/elo/qb): how much does the market move per second?
BASE = ["elo_sum", "pf_sum", "pa_sum", "qb_sum", "dome", "wind_f", "div"]
for lab, x in [("train", tr), ("test", te)]:
    X = sm.add_constant(x[BASE + ["spp_avg"]].astype(float)); f = sm.OLS(x.mkt_total - x.lg_blend, X).fit(cov_type="HC1")
    print(f"  market total ~ BASE + spp_avg [{lab}]: market slope on spp_avg = {f.params['spp_avg']:+.3f} (se {f.bse['spp_avg']:.3f}, p={f.pvalues['spp_avg']:.3f})")
# implied FF / SS adjustment from the 2009-2019 market-free fit (b = -0.541 from 02/05)
b = -0.541
lo_q, hi_q = np.percentile(pd.concat([tr.h_spp_neut_r8d, tr.a_spp_neut_r8d]), [25, 75])
for lab, x in [("train", tr), ("test", te)]:
    ff = x[(x.h_spp_neut_r8d <= lo_q) & (x.a_spp_neut_r8d <= lo_q)]; ss = x[(x.h_spp_neut_r8d >= hi_q) & (x.a_spp_neut_r8d >= hi_q)]
    print(f"  {lab}: FF mean spp_avg={ff.spp_avg.mean():+.2f} s -> implied adj {b*ff.spp_avg.mean():+.2f} pts (n={len(ff)}); SS mean {ss.spp_avg.mean():+.2f} s -> {b*ss.spp_avg.mean():+.2f} pts (n={len(ss)})")
# season-level persistence of team pace (prior-8 vs realized in that game)
tg = pd.read_csv(HERE / "_teamgame_feats.csv", low_memory=False)
tg = tg[tg.game_type == "REG"].dropna(subset=["spp_neut_r8", "spp_neut"])
for era, x in [("2009-19", tg[tg.season <= 2019]), ("2023-25", tg[tg.season >= 2023])]:
    print(f"  team level {era}: corr(prior-8 spp, this-game spp)={stats.pearsonr(x.spp_neut_r8, x.spp_neut)[0]:+.3f}; SD of prior-8 spp={x.spp_neut_r8.std():.2f}; SD of game spp={x.spp_neut.std():.2f}; n={len(x)}")
