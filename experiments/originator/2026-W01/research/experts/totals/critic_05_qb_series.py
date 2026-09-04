"""CRITIC 05: is the 538/nfelo QB adjustment (qb_sum) a live, pre-game series across all seasons incl. 2024-2026?
The rolling ablation shows the QB term HURTING in 2025 (-0.119). Check coverage, dispersion, leakage vs market."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from kit import load_nfelo
from critic_common import build_fixed
n = load_nfelo()
n["qb_sum"] = (n.home_538_qb_adj.fillna(0) + n.away_538_qb_adj.fillna(0)) / 25
print("538 QB adj by season (nfelo_games.csv): coverage, share exactly 0, SD of home adj, mean|adj|")
g = n.groupby("season").agg(n=("qb_sum", "size"), cov=("home_538_qb_adj", lambda s: s.notna().mean()), zero=("home_538_qb_adj", lambda s: (s.fillna(0) == 0).mean()),
                            sd=("home_538_qb_adj", "std"), mabs=("home_538_qb_adj", lambda s: s.abs().mean()), sd_sum=("qb_sum", "std"))
print(g.round(3).to_string())
m = build_fixed(K_team=1, K_lg=128)
d = m[(m.game_type == "REG") & m.elo_sum.notna()].copy()
print("\nleak check: market residual (total - mkt) ~ qb_sum, and mkt_total ~ qb_sum (does the market price it?)")
for lab, s in [("2009-2021", d.train), ("2022-2025", d.test), ("2024-2025", d.season >= 2024)]:
    x = d[s]; f1 = smf.ols("total_err_mkt ~ qb_sum", data=x).fit(cov_type="HC1"); f2 = smf.ols("mkt_total ~ qb_sum + elo_sum", data=x).fit(cov_type="HC1")
    f3 = smf.ols("total_pts ~ qb_sum + elo_sum + pf_dev + pa_dev", data=x).fit(cov_type="HC1")
    print(f"  {lab}: resid~qb_sum {f1.params['qb_sum']:+.3f} (p={f1.pvalues['qb_sum']:.2f}) | mkt~qb_sum {f2.params['qb_sum']:+.3f} (se {f2.bse['qb_sum']:.3f}) | total~qb_sum (with elo,pf,pa) {f3.params['qb_sum']:+.3f} (se {f3.bse['qb_sum']:.3f}) n={len(x)}")
print("\nper-season in-sample qb_sum coefficient (total ~ qb_sum + elo_sum + pf_dev + pa_dev):")
for Y in range(2009, 2026):
    x = d[d.season == Y]; f = smf.ols("total_pts ~ qb_sum + elo_sum + pf_dev + pa_dev", data=x).fit(cov_type="HC1")
    print(f"  {Y}: {f.params['qb_sum']:+.2f} (se {f.bse['qb_sum']:.2f})  sd(qb_sum)={x.qb_sum.std():.2f}")
