"""Synthesis check 04: confidence-tag bases (uncertainty critic: 5-season RMSE) and distance-to-market bands
on the UNREGRESSED nfelo line (-nfelo_dif_base/25) vs the closing spread, 2022-25 REG."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from kit import load_games, load_nfelo
g = load_games(); n = load_nfelo()
m = g.merge(n[["gid", "nfelo_dif_base", "nfelo_home_line_close"]], on="gid", how="inner")
m = m[(m.game_type == "REG") & m.mkt_spread.notna() & m.mkt_total.notna()].copy()
for win in [(2021, 2025), (2023, 2025), (2019, 2025)]:
    w = m[(m.season >= win[0]) & (m.season <= win[1])]
    print(f"[A] BASE {win}: spread RMSE(margin+close) {np.sqrt((w.spread_err_mkt**2).mean()):.2f} | total RMSE {np.sqrt((w.total_err_mkt**2).mean()):.2f} (n={len(w)})")
print("[A] per-season spread RMSE:", m.groupby("season").spread_err_mkt.apply(lambda s: np.sqrt((s**2).mean())).round(2).to_dict())
m["model"] = -m.nfelo_dif_base / 25.0           # ORIGINATOR sign
m["D"] = (m.model - m.mkt_spread).abs()
m["e_model"] = m.margin + m.model; m["e_mkt"] = m.spread_err_mkt
t = m[m.season >= 2022]
base = np.sqrt((m[(m.season >= 2021) & (m.season <= 2025)].e_mkt**2).mean())
print(f"[B] 2022-25 n={len(t)}: model (unregressed nfelo) MAE {t.e_model.abs().mean():.3f} vs market {t.e_mkt.abs().mean():.3f}; mean D {t.D.mean():.2f}; corr(e_mkt, model-mkt) {np.corrcoef(t.e_mkt, t.model - t.mkt_spread)[0,1]:+.3f}")
for cuts in [(1.5, 3.0), (1.75, 3.5), (2.0, 4.0)]:
    band = pd.cut(t.D, [-1, cuts[0], cuts[1], 99], labels=["HIGH", "MED", "LOW"])
    print(f"[B] bands {cuts}: ", end="")
    for b in ["HIGH", "MED", "LOW"]:
        s = t[band == b]
        rm_model = np.sqrt((s.e_model**2).mean()); rm_mkt = np.sqrt((s.e_mkt**2).mean())
        ident = np.sqrt(base**2 + (s.D**2).mean())
        print(f"{b}: share {len(s)/len(t):.2f} model RMSE {rm_model:.2f} mkt RMSE {rm_mkt:.2f} excess {rm_model-rm_mkt:+.2f} identity-pred {ident-base:+.2f} | ", end="")
    print()
# disagreement does not predict MARKET error size
import statsmodels.api as sm
f = sm.OLS(t.e_mkt.abs(), sm.add_constant(t.D)).fit(cov_type="HC1")
print(f"[C] |market err| ~ D (2022-25): slope {f.params.D:+.3f} (se {f.bse.D:.3f}, p={f.pvalues.D:.2f}); Spearman {t[['D']].assign(a=t.e_mkt.abs()).corr('spearman').iloc[0,1]:+.3f}")
f2 = sm.OLS(t.e_model.abs(), sm.add_constant(t.D)).fit(cov_type="HC1")
print(f"[C] |model err| ~ D (2022-25): slope {f2.params.D:+.3f} (se {f2.bse.D:.3f}, p={f2.pvalues.D:.3f})")
