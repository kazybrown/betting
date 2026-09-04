"""Synthesis check 01: sign convention + the kit.py join hole reported by the totals critic.
Run from anywhere. Prints every number cited in the synthesis."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import load_games, load_nfelo, norm, merged

g = load_games()
n = load_nfelo()
reg = g[(g.game_type == "REG") & g.mkt_spread.notna()]
print("sign check: corr(mkt_spread, margin) =", round(np.corrcoef(reg.mkt_spread, reg.margin)[0, 1], 3), "(must be strongly negative)")
print("sign check: corr(mkt_total, total_pts) =", round(np.corrcoef(reg.mkt_total, reg.total_pts)[0, 1], 3), "(positive)")

# kit join (gid rewrites only _LAR_/_OAK_) vs fully normalized join
m_kit = merged()
parts = g.game_id.str.split("_", expand=True)
g["gid_fix"] = parts[0] + "_" + parts[1] + "_" + parts[2].map(norm) + "_" + parts[3].map(norm)
m_fix = g.merge(n[["gid", "starting_nfelo_home"]].rename(columns={"gid": "gid_fix"}), on="gid_fix", how="left")
for lab, m in [("kit.merged()", m_kit), ("normalized both parts", m_fix)]:
    r = m[m.game_type == "REG"]
    cov = r.starting_nfelo_home.notna()
    print(f"{lab}: REG 2009-25 nfelo coverage {cov.mean():.3f}  missing={int((~cov).sum())}  "
          f"2009-2019 missing={int((~cov[r.season<=2019]).sum())}  2022-25 missing={int((~cov[r.season>=2022]).sum())}")
miss = m_kit[(m_kit.game_type == "REG") & m_kit.starting_nfelo_home.isna()]
print("kit-missing games by season:", miss.groupby("season").size().to_dict())
print("kit-missing games involving LA/LAC/LV ids:", int(miss.game_id.str.contains("STL|SD|LAC|LA_|OAK|LV").sum()), "of", len(miss))
