"""CRITIC 00: data sanity checks on the calibration expert's frame (common.load()).
- sign conventions (corr with margin) for every line and for the QB / Elo terms
- nfelo's market column sign errors vs the nflverse close
- reconstruction of nfelo's own unregressed line (nfelo_own): implied Elo/pt, root ambiguity
- live nfelo (historic_projected_spreads.csv) mapping: is pre-regression line = -dif/25 ?
Run from this directory: python3 critic_00_data_checks.py
"""
import numpy as np, pandas as pd
from common import load, D
pd.set_option("display.width", 200)
m = load(verbose=True)

print("\n== sign checks: corr(x, margin) ==")
for c in ["mkt", "mkt_nfelo", "nfelo_close", "nfelo_lin", "nfelo_own", "qbelo", "nfelo_noqb"]:
    print(f"  {c:12s} {np.corrcoef(m[c], m.margin)[0,1]:+.3f}  (a line: must be negative)")
print(f"  qb_pts       {np.corrcoef(m.qb_pts, m.margin)[0,1]:+.3f}  (home QB better -> home wins more: must be positive)")
print(f"  elo_dif_pts  {np.corrcoef(m.elo_dif_pts, m.margin)[0,1]:+.3f}  (must be positive)")
print(f"  hfa_pts      mean {m.hfa_pts.mean():.2f} pts (must be positive, ~1.5-2.7)")

bad = m[(np.sign(m.mkt_nfelo) != np.sign(m.mkt)) & (m.mkt != 0) & (m.mkt_nfelo != 0)]
print(f"\nnfelo market column: sign disagreements with nflverse close = {len(bad)} of {len(m)}; |diff|>3 pts = {int((np.abs(m.mkt_nfelo-m.mkt)>3).sum())}; |diff|<=0.5 = {(np.abs(m.mkt_nfelo-m.mkt)<=0.5).mean():.3f}")
print(bad[["gid", "mkt", "mkt_nfelo"]].head(8).to_string(index=False))

r = np.sqrt(m.nfelo_unreg_se)
c1, c2 = -m.margin + r, -m.margin - r
amb = (np.abs(np.abs(c1 - m.nfelo_lin) - np.abs(c2 - m.nfelo_lin)) < 1.0)
print(f"\nnfelo_own root reconstruction: ambiguous (<1 pt separation between roots' distance to nfelo_lin) in {amb.mean():.3%} of games; sqrt(se)<1 in {(r<1).mean():.3%}")
ratio = m.nfelo_dif_base / (-m.nfelo_own)
k = (m.nfelo_own.abs() > 2) & ~amb
print(f"implied Elo/pt of nfelo's own unregressed line (nfelo_dif_base / -nfelo_own), |line|>2 & unambiguous (n={int(k.sum())}): "
      f"median {ratio[k].median():.2f} IQR [{ratio[k].quantile(.25):.2f},{ratio[k].quantile(.75):.2f}]")
print("  by season (median):", ratio[k].groupby(m.season[k]).median().round(2).to_dict())
print(f"  corr(nfelo_own, nfelo_lin) = {np.corrcoef(m.nfelo_own, m.nfelo_lin)[0,1]:.4f}; mean|nfelo_own - nfelo_lin| = {np.abs(m.nfelo_own - m.nfelo_lin).mean():.3f}; "
      f"share |diff|<0.5: {(np.abs(m.nfelo_own - m.nfelo_lin) < 0.5).mean():.3f}")
# is nfelo_own just nfelo_lin rounded to 0.5?
rnd = np.round(m.nfelo_lin * 2) / 2
print(f"  share nfelo_own == round(nfelo_lin, 0.5): {(np.abs(m.nfelo_own - rnd) < 0.01).mean():.3f}; share nfelo_own is a multiple of 0.5: {(np.abs(m.nfelo_own*2 - np.round(m.nfelo_own*2)) < 0.01).mean():.3f}")

h = pd.read_csv(D / "historic_projected_spreads.csv", low_memory=False)
d = h.home_line_pre_regression + h.home_dif_pre_reg / 25
print(f"\nlive nfelo (historic file, n={len(h)}): home_line_pre_regression == -home_dif_pre_reg/25 within 0.01 in {(np.abs(d) < 0.01).mean():.3%} of rows "
      f"-> live nfelo maps Elo to points at 25 Elo/pt pre-regression")
comp = (h.home_nfelo_elo - h.away_nfelo_elo) + h.home_net_HFA_mod.fillna(0) + h.home_net_bye_mod.fillna(0) + h.home_net_qb_mod.fillna(0)
ok = np.abs(comp - h.home_dif_pre_reg) < 1
print(f"live decomposition home_dif_pre_reg == elo_dif + HFA + bye + qb within 1 Elo: {ok.mean():.3%} (failures by version: {h.loc[~ok, 'nfelo_version'].value_counts().to_dict()})")
