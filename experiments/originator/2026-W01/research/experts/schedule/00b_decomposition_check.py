"""00b: verify how nfelo_dif_base decomposes so the 'rating-only' line strips exactly the bye + tz mods."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import load_nfelo
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
n = load_nfelo().dropna(subset=["nfelo_dif_base"])
n["season"] = n.season.astype(int)
base = n.starting_nfelo_home - n.starting_nfelo_away
mods = dict(hfa=n.hfa_mod, qb=n.home_net_qb_mod.fillna(0), div=n.div_game_mod.fillna(0), surf=n.dif_surface_mod.fillna(0),
            tz=n.home_time_advantage_mod.fillna(0), hbye=n.home_bye_mod.fillna(0), abye=n.away_bye_mod.fillna(0),
            netbye=n.home_net_bye_mod.fillna(0))
def check(label, s):
    r = n.nfelo_dif_base - s
    print("%-45s mean %+7.3f sd %7.3f max|.| %7.3f  (rows |r|>0.5: %d)" % (label, r.mean(), r.std(), r.abs().max(), (r.abs()>0.5).sum()))
    return r
check("base+hfa+qb+div+surf+tz+hbye+abye", base+mods["hfa"]+mods["qb"]+mods["div"]+mods["surf"]+mods["tz"]+mods["hbye"]+mods["abye"])
check("base+hfa+qb+div+surf+tz+netbye", base+mods["hfa"]+mods["qb"]+mods["div"]+mods["surf"]+mods["tz"]+mods["netbye"])
r = check("base+hfa+qb+div+surf+hbye+abye (no tz)", base+mods["hfa"]+mods["qb"]+mods["div"]+mods["surf"]+mods["hbye"]+mods["abye"])
check("base+hfa_base+qb+div+surf+tz+hbye+abye", base+n.hfa_base_mod.fillna(0)+mods["qb"]+mods["div"]+mods["surf"]+mods["tz"]+mods["hbye"]+mods["abye"])
print("\nhome_net_bye_mod vs home_bye_mod+away_bye_mod: corr=%.3f; mean diff=%.3f" % (np.corrcoef(mods["netbye"], mods["hbye"]+mods["abye"])[0,1], (mods["netbye"]-mods["hbye"]-mods["abye"]).mean()))
print("rows where net bye != hbye+abye:", ((mods["netbye"]-mods["hbye"]-mods["abye"]).abs()>0.01).sum())
big = n[(r.abs()>0.5)]
print("\nrows with large residual (no-tz formula) by season:", big.season.value_counts().sort_index().to_dict())
print(big[["game_id","hfa_mod","hfa_base_mod","home_time_advantage_mod","home_bye_mod","away_bye_mod","home_net_bye_mod","home_net_qb_mod","nfelo_dif_base"]].head(8))
# Is hfa_mod == hfa_base_mod + tz in some seasons?
d = (n.hfa_mod - n.hfa_base_mod.fillna(0) - n.home_time_advantage_mod.fillna(0))
print("\nhfa_mod - hfa_base - tz by season (mean, sd):"); print(pd.DataFrame({"mean": d.groupby(n.season).mean(), "sd": d.groupby(n.season).std()}).round(2).T)
d2 = (n.hfa_mod - n.hfa_base_mod.fillna(0))
print("hfa_mod - hfa_base by season (mean, sd):"); print(pd.DataFrame({"mean": d2.groupby(n.season).mean(), "sd": d2.groupby(n.season).std()}).round(2).T)
