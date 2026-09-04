"""critic_01_frame_checks.py - CRITIC: frame integrity for the uncertainty expert.
 (a) sign checks (README rule 5) done fresh;
 (b) nflverse closing spread vs nfelo's own market close: share identical, |diff| distribution, which one
     the market-regressed nfelo actually tracks, and which one has the lower error;
 (c) what is nfelo_dif_base? reconstruct it from components; compare to home_line_pre_regression;
 (d) an INDEPENDENT engine (538 qbelo, never market-informed): |qbelo - mkt| distribution and band shares
     vs nfelo_b -> does the 54/30/16 band split transfer to an engine that is not regressed to the market?
Run: cd /home/user/originator-2026-w01/research && python3 experts/uncertainty/critic_01_frame_checks.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/uncertainty")
from common import build
from kit import load_games

pd.set_option("display.width", 200)
m = build()
print(f"frame n={len(m)} seasons {m.season.min()}-{m.season.max()} fit={int((m.era=='fit').sum())} test={int((m.era=='test').sum())}")
g = load_games()
g = g[(g.season >= 2009) & g.mkt_spread.notna()]
print(f"nflverse scored games 2009-25 with a closing spread: {len(g)} -> dropped by the nfelo join: {len(g) - len(m)}")
print("dropped game types:", g[~g.gid.isin(m.gid)].game_type.value_counts().to_dict())

print("\n(a) SIGN CHECKS")
for c in ["mkt", "nfelo_b", "nfelo_c", "home_line_close", "mkt_open"]:
    x = m[[c, "margin"]].dropna(); print(f"  corr({c}, margin) = {np.corrcoef(x[c], x.margin)[0,1]:+.3f}")
print(f"  mean e_mkt (margin+mkt) = {m.e_mkt.mean():+.3f} (should be ~0)")

print("\n(b) nflverse close vs nfelo market close")
d = (m.mkt - m.home_line_close)
print(f"  identical share {(d.abs()<0.01).mean():.3f} | mean |diff| {d.abs().mean():.3f} | p90 |diff| {d.abs().quantile(.9):.2f} | max {d.abs().max():.1f} | mean signed diff {d.mean():+.3f}")
print("  |diff| by season:", m.assign(ad=d.abs()).groupby('season').ad.mean().round(2).to_dict())
e_nfeloclose = m.margin + m.home_line_close
print(f"  RMSE vs results: nflverse close {np.sqrt((m.e_mkt**2).mean()):.3f} | nfelo market close {np.sqrt((e_nfeloclose**2).mean()):.3f} | MAE {m.ae_mkt.mean():.3f} vs {e_nfeloclose.abs().mean():.3f}")
print(f"  nfelo regressed line == nfelo market close share: {((m.nfelo_c - m.home_line_close).abs()<0.01).mean():.3f} | == nflverse close share: {((m.nfelo_c - m.mkt).abs()<0.01).mean():.3f}")
# D_base against the two closes
D2 = (m.nfelo_b - m.home_line_close).abs()
print(f"  D_base vs nflverse close: mean {m.D_base.mean():.3f} | vs nfelo market close: mean {D2.mean():.3f} | corr {np.corrcoef(m.D_base, D2)[0,1]:.3f}")

print("\n(c) what is nfelo_dif_base?")
n = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_games.csv", low_memory=False)
n["elo_dif"] = n.starting_nfelo_home - n.starting_nfelo_away
cands = {
  "elo_dif + hfa_mod + div + surf + time + qb + bye": n.elo_dif + n.hfa_mod + n.div_game_mod.fillna(0) + n.dif_surface_mod.fillna(0) + n.home_time_advantage_mod.fillna(0) + n.home_net_qb_mod.fillna(0) + n.home_net_bye_mod.fillna(0),
  "elo_dif + hfa_base_mod + div + surf + time + qb + bye": n.elo_dif + n.hfa_base_mod.fillna(0) + n.div_game_mod.fillna(0) + n.dif_surface_mod.fillna(0) + n.home_time_advantage_mod.fillna(0) + n.home_net_qb_mod.fillna(0) + n.home_net_bye_mod.fillna(0),
  "elo_dif + hfa_mod + qb + bye": n.elo_dif + n.hfa_mod + n.home_net_qb_mod.fillna(0) + n.home_net_bye_mod.fillna(0),
  "elo_dif + hfa_mod + qb": n.elo_dif + n.hfa_mod + n.home_net_qb_mod.fillna(0),
}
for k, v in cands.items():
    r = (n.nfelo_dif_base - v); print(f"  nfelo_dif_base - [{k}]: mean|.|={r.abs().mean():.2f} Elo, exact(<0.5) share={(r.abs()<0.5).mean():.3f}")
print("  hfa_base_mod non-null share by season:", n.assign(s=n.game_id.str[:4]).groupby('s').hfa_base_mod.apply(lambda x: x.notna().mean()).round(2).to_dict())
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
j = m.merge(h[["gid", "home_line_pre_regression", "home_dif_pre_reg", "home_closing_line_rounded_nfelo", "market_regression_factor", "home_line_close"]].rename(columns={"home_line_close": "h_close"}), on="gid", how="inner")
gap = j.nfelo_b - j.home_line_pre_regression
print(f"  joined to historic file n={len(j)}: nfelo_b - home_line_pre_regression: mean {gap.mean():+.3f} sd {gap.std():.3f} | corr {np.corrcoef(j.nfelo_b, j.home_line_pre_regression)[0,1]:.4f}")
print(f"  |gap| by season: {j.assign(a=gap.abs()).groupby('season').a.mean().round(2).to_dict()}")
print(f"  D from historic pre-regression line vs its own close: mean {(j.home_line_pre_regression - j.h_close).abs().mean():.3f} | D_base (nfelo_b vs nflverse close): mean {j.D_base.mean():.3f} | corr of the two D's {np.corrcoef((j.home_line_pre_regression - j.h_close).abs(), j.D_base)[0,1]:.3f}")
print(f"  market_regression_factor: mean {j.market_regression_factor.mean():.3f} sd {j.market_regression_factor.std():.3f}")

print("\n(d) INDEPENDENT engine: 538 qbelo line vs market")
s = pd.read_csv("/home/user/originator-2026-w01/research/data/nfelo_scored_individual_games.csv", low_memory=False)
nn = n.iloc[: len(s)].reset_index(drop=True)
s["gid"] = nn.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_").values
q = m.merge(s[["gid", "qbelo_home_line_close_rounded"]].rename(columns={"qbelo_home_line_close_rounded": "qbelo"}), on="gid", how="inner").dropna(subset=["qbelo"])
q["D_q"] = (q.qbelo - q.mkt).abs(); q["e_q"] = q.margin + q.qbelo
print(f"  n={len(q)} corr(qbelo, margin)={np.corrcoef(q.qbelo, q.margin)[0,1]:+.3f} | MAE qbelo {q.e_q.abs().mean():.3f} vs market {q.ae_mkt.mean():.3f} vs nfelo_b {q.ae_nb.mean():.3f}")
for lab, col in [("nfelo_b (market-informed Elo)", "D_base"), ("qbelo (independent Elo)", "D_q")]:
    x = q[col]; t = q[q.era == "test"][col]
    print(f"  {lab:32s} D mean {x.mean():.2f} median {x.median():.2f} p90 {x.quantile(.9):.2f} | test-era band shares HIGH<1.5 {(t<1.5).mean():.2f} MED {((t>=1.5)&(t<3)).mean():.2f} LOW>=3 {(t>=3).mean():.2f}")
print("  -> if ORIGINATOR's PFF/Cole engines are independent of the market like qbelo, the LOW band share will be far larger than the 16% quoted for nfelo.")
