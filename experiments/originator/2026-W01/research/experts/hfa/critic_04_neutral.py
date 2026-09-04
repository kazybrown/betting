"""CRITIC HFA-4: neutral/international. (a) recount including the 4 Bills-in-Toronto games coded location==Home;
(b) how nfelo's hfa_mod / tz behave at neutral sites; (c) bootstrap CIs for the OOS (n=22) k comparison; (d) London-only."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from scipy import stats
from kit import merged, load_nfelo
pd.set_option("display.width", 220)
rng = np.random.default_rng(4)
m = merged()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0))/25
m["r"] = m.margin - m.rating_dif; m["mkt_hfa"] = m.spread_line - m.rating_dif
INTL_STAD = ["Wembley Stadium","Twickenham Stadium","Tottenham Stadium","Azteca Stadium","Allianz Arena","Deutsche Bank Park","Arena Corinthians","Rogers Centre"]
INTL_2025 = ["2025_01_KC_LAC","2025_04_MIN_PIT","2025_05_MIN_CLE","2025_06_DEN_NYJ","2025_07_LA_JAX","2025_10_ATL_IND","2025_11_WAS_MIA"]
m["intl"] = m.game_type.eq("REG") & (m.stadium.isin(INTL_STAD) | m.gid.isin(INTL_2025))
d = m[m.intl]
print("(a) international REG games by location flag:", d.location.value_counts().to_dict(), "| rated:", d.r.notna().sum())
print("    location==Home international games (Toronto):"); print(d[d.location.eq("Home")][["season","week","away","home","stadium","spread_line","result","r","spread_err_mkt","hfa_pts"]].round(2).to_string(index=False))
for lab, dd in [("expert set (Neutral only)", d[d.location.eq("Neutral")]), ("incl. Toronto Home games", d), ("ex Toronto entirely", d[~d.stadium.eq("Rogers Centre")])]:
    x = dd.r.dropna().values; bs = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    y = dd.spread_err_mkt.dropna().values
    print("    %-26s n=%d rated: edge=%.2f  CI[%.2f, %.2f]  median=%.2f | market resid %+.2f (se %.2f) home cover %d-%d" %
          (lab, len(x), x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5), np.median(x), y.mean(), y.std()/np.sqrt(len(y)), (y>0).sum(), (y<0).sum()))
# (b) nfelo at neutral sites
n_ = load_nfelo()
neu = m[m.neutral & m.game_type.eq("REG")].merge(n_[["gid","hfa_base_mod"]], on="gid", how="left")
neu["tz"] = neu.home_time_advantage_mod.fillna(0)/25; neu["base"] = neu.hfa_base_mod/25; neu["mod"] = neu.hfa_mod/25
print("\n(b) nfelo components at neutral REG sites (pts): |hfa_mod/25|>0.25 in %d/%d; |hfa_base_mod/25|>0.25 in %d/%d; |tz|>0.1 in %d/%d" %
      ((neu["mod"].abs()>0.25).sum(), neu["mod"].notna().sum(), (neu.base.abs()>0.25).sum(), neu.base.notna().sum(), (neu.tz.abs()>0.1).sum(), neu.tz.notna().sum()))
print("    by season: mean |hfa_mod/25| =", neu.groupby("season")["mod"].apply(lambda s: s.abs().mean()).round(2).to_dict())
print("    2021-25 neutral games with nonzero hfa:"); print(neu[(neu.season>=2021)&(neu["mod"].abs()>0.1)][["season","week","away","home","stadium","mod","base","tz","hfa_pts"]].round(2).to_string(index=False))
# is nfelo's neutral HFA correlated with outcome at all?
z = neu.dropna(subset=["r","hfa_pts"]); print("    corr(nfelo hfa_pts, r) at neutral sites: %.2f (n=%d)" % (np.corrcoef(z.hfa_pts, z.r)[0,1], len(z)))
# (c) OOS n=22 bootstrap of MAE differences
te = d[(d.season>=2022)&d.r.notna()]; n = len(te); idx = rng.integers(0, n, (5000, n))
print("\n(c) OOS 2022-25 international (n=%d): paired-bootstrap MAE differences vs k=0.75" % n)
e75 = np.abs(te.r.values - 0.75)
for k in [0.0, 2.0]:
    dd = np.abs(te.r.values - k) - e75; bs = dd[idx].mean(axis=1)
    print("    MAE(k=%.2f) - MAE(k=0.75) = %+.3f  CI[%+.3f, %+.3f]" % (k, dd.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)))
# (d) London only, and by home-team continent-of-origin irrelevant; check the fit/test split
lon = d[d.stadium.isin(["Wembley Stadium","Twickenham Stadium","Tottenham Stadium"]) | d.gid.isin(["2025_05_MIN_CLE","2025_06_DEN_NYJ","2025_07_LA_JAX"])]
x = lon.r.dropna().values; print("\n(d) London n=%d: edge %.2f (se %.2f), market resid %+.2f, home cover %d-%d" % (len(x), x.mean(), x.std()/np.sqrt(len(x)), lon.spread_err_mkt.mean(), (lon.spread_err_mkt>0).sum(), (lon.spread_err_mkt<0).sum()))
