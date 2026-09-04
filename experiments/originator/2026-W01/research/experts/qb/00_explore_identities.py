"""00: verify nfelo column identities + QB-name coverage. Run from anywhere."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, load_nfelo, merged

n = load_nfelo()
print("nfelo rows:", len(n), "seasons", n.season.min(), "-", n.season.max())
mods = ["home_bye_mod","away_bye_mod","div_game_mod","dif_surface_mod","home_time_advantage_mod","home_net_qb_mod"]
elo_dif = n.starting_nfelo_home - n.starting_nfelo_away
recon_with_qb = elo_dif + n.hfa_base_mod + n[mods].fillna(0).sum(axis=1)
recon_no_qb   = elo_dif + n.hfa_base_mod + n[mods[:-1]].fillna(0).sum(axis=1)
recon_hfa_mod = elo_dif + n.hfa_mod + n[mods].fillna(0).sum(axis=1)
for name, r in [("elo_dif+hfa_base+all mods incl QB", recon_with_qb),
                ("elo_dif+hfa_base+mods EXCL QB", recon_no_qb),
                ("elo_dif+hfa_mod+all mods incl QB", recon_hfa_mod)]:
    d = (n.nfelo_dif_base - r)
    print(f"  identity check nfelo_dif_base == {name}: median|diff|={d.abs().median():.4f}, "
          f"share within 0.01 = {(d.abs()<0.01).mean():.3f}, mean diff={d.mean():.3f}")
# hfa_mod vs hfa_base_mod + time
d2 = n.hfa_mod - (n.hfa_base_mod + n.home_time_advantage_mod.fillna(0))
print("hfa_mod == hfa_base_mod + home_time_advantage_mod ? share within .01:", (d2.abs()<0.01).mean().round(3))
print("home_net_qb_mod == home_538_qb_adj - away_538_qb_adj ? share within .01:",
      ((n.home_net_qb_mod - (n.home_538_qb_adj - n.away_538_qb_adj)).abs()<0.01).mean().round(3))
print("\n538 QB adj (Elo pts) describe (home):"); print(n.home_538_qb_adj.describe().round(2))
print("home_net_qb_mod / 25 -> points describe:"); print((n.home_net_qb_mod/25).describe().round(3))
print("by season mean |home_538_qb_adj|/25:")
print((n.assign(a=(n.home_538_qb_adj.abs()/25)).groupby("season").a.mean()).round(3).to_dict())

# nfelo_dif_open/close vs base: regression toward market?
n["mkt_open_elo"] = -n.home_line_open*25; n["mkt_close_elo"] = -n.home_line_close*25
sub = n.dropna(subset=["home_line_close","nfelo_dif_close"])
frac = ((sub.nfelo_dif_close - sub.nfelo_dif_base)/(sub.mkt_close_elo - sub.nfelo_dif_base)).replace([np.inf,-np.inf],np.nan)
print("\nimplied regression fraction toward market close, median:", frac.median().round(3), " IQR:", frac.quantile([.25,.75]).round(3).tolist())
print("corr(nfelo_home_line_close, -nfelo_dif_close/25):", np.corrcoef(sub.nfelo_home_line_close, -sub.nfelo_dif_close/25)[0,1].round(4))

g = load_games(2009)
print("\ngames rows 2009+:", len(g))
print("QB name missing share by season (home):", g.groupby("season").home_qb_name.apply(lambda s: s.isna().mean()).round(3).to_dict())
print("game_type counts:", g.game_type.value_counts().to_dict())
m = merged(2009)
print("merged with nfelo (non-null nfelo_dif close) :", m.nfelo_home_line_close.notna().sum())
print("sign check corr(mkt_spread, margin):", np.corrcoef(m.dropna(subset=['mkt_spread']).mkt_spread, m.dropna(subset=['mkt_spread']).margin)[0,1].round(3))
