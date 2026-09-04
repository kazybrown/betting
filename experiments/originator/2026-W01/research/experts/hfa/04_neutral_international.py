"""THEORY 4: Neutral-site and international games: realized edge for the designated home team.
International = location Neutral & REG & (overseas stadium) or the 2025 overseas slate (nflverse lists the
home team's own stadium for 2025 neutral games, so those are identified by game id).
Domestic neutral = other REG Neutral (relocations: Ford Field, State Farm 2020, Jacksonville 2021 ...)."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd
from scipy import stats
from kit import merged, load_games
pd.set_option("display.width", 220)
m = merged()
m["rating_dif"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod.fillna(0)) / 25.0
m["hfa_resid"] = m.margin - m.rating_dif
m["mkt_hfa"] = m.spread_line - m.rating_dif
INTL_STAD = ["Wembley Stadium","Twickenham Stadium","Tottenham Stadium","Azteca Stadium","Allianz Arena","Deutsche Bank Park","Arena Corinthians"]
INTL_2025 = {"2025_01_KC_LAC":"Sao Paulo","2025_04_MIN_PIT":"Dublin","2025_05_MIN_CLE":"London","2025_06_DEN_NYJ":"London",
             "2025_07_LA_JAX":"London","2025_10_ATL_IND":"Berlin","2025_11_WAS_MIA":"Madrid"}
def venue(r):
    if r.gid in INTL_2025: return INTL_2025[r.gid]
    s = r.stadium
    if s in ["Wembley Stadium","Twickenham Stadium","Tottenham Stadium"]: return "London"
    if s == "Azteca Stadium": return "Mexico"
    if s in ["Allianz Arena","Deutsche Bank Park"]: return "Germany"
    if s == "Arena Corinthians": return "Sao Paulo"
    if s == "Rogers Centre": return "Toronto"
    return "domestic"
neu = m[m.neutral].copy(); neu["venue"] = neu.apply(venue, axis=1)
neu["intl"] = neu.venue.ne("domestic") & neu.game_type.eq("REG")
neu["cover_home"] = np.sign(neu.spread_err_mkt)
print("Neutral games by class:", neu.groupby([neu.game_type.eq("REG").map({True:"REG",False:"POST"}), "venue"]).size().to_dict())

def summarize(d, label):
    d = d.copy(); n = len(d); nr = d.hfa_resid.notna().sum()
    mk = d.spread_err_mkt.dropna()
    out = dict(label=label, n=n, n_rated=nr, raw_margin=d.margin.mean(), raw_se=d.margin.std()/np.sqrt(n),
               rating_hfa=d.hfa_resid.mean(), rating_hfa_se=d.hfa_resid.std()/np.sqrt(max(nr,1)),
               mkt_impl_edge=d.mkt_hfa.mean(), mean_spread_line=d.spread_line.mean(),
               mkt_resid=mk.mean(), mkt_resid_se=mk.std()/np.sqrt(len(mk)), home_cover=(mk>0).sum()/max((mk!=0).sum(),1),
               nfelo_hfa=d.hfa_pts.mean(), nfelo_abs_hfa=d.hfa_pts.abs().mean())
    return out
rows = [summarize(neu[neu.intl], "INTL all (REG)"),
        summarize(neu[neu.intl & (neu.season<=2021)], "INTL fit <=2021"),
        summarize(neu[neu.intl & (neu.season>=2022)], "INTL test 2022-25"),
        summarize(neu[neu.intl & neu.home.eq("JAX")], "INTL JAX home"),
        summarize(neu[neu.intl & ~neu.home.eq("JAX")], "INTL non-JAX home"),
        summarize(neu[neu.venue.eq("London")], "London"),
        summarize(neu[neu.venue.eq("Germany")], "Germany"),
        summarize(neu[neu.venue.eq("Mexico")], "Mexico"),
        summarize(neu[neu.venue.isin(["Sao Paulo","Dublin","Madrid","Berlin"])], "Brazil/Ireland/Spain/Berlin"),
        summarize(neu[~neu.intl & neu.game_type.eq("REG")], "Domestic neutral REG"),
        summarize(neu[neu.game_type.eq("SB")], "Super Bowls"),
        summarize(m[~m.neutral & m.game_type.eq("REG") & (m.season>=2009) & (m.season!=2020)], "REF: home games REG 2009-25 ex2020"),
        summarize(m[~m.neutral & m.game_type.eq("REG") & (m.season>=2021)], "REF: home games REG 2021-25")]
T = pd.DataFrame(rows).set_index("label"); print("\n(points; designated-home perspective)"); print(T.round(2).to_string())
# bootstrap CI for intl rating HFA and market residual
rng = np.random.default_rng(11)
d = neu[neu.intl]; x = d.hfa_resid.dropna().values; y = d.spread_err_mkt.dropna().values
bx = [rng.choice(x, len(x)).mean() for _ in range(5000)]; by = [rng.choice(y, len(y)).mean() for _ in range(5000)]
print("\nINTL (REG) designated-home rating-adjusted edge: %.2f  95%% bootstrap CI [%.2f, %.2f] (n=%d)" % (x.mean(), np.percentile(bx,2.5), np.percentile(bx,97.5), len(x)))
print("INTL market residual (+ = designated home under-priced): %.2f  CI [%.2f, %.2f] (n=%d); home cover %d-%d-%d" %
      (y.mean(), np.percentile(by,2.5), np.percentile(by,97.5), len(y), (y>0).sum(), (y<0).sum(), (y==0).sum()))
# Welch vs league home games same seasons
ref = m[~m.neutral & m.game_type.eq("REG") & (m.season!=2020) & m.season.isin(d.season.unique())]
tt = stats.ttest_ind(x, ref.hfa_resid.dropna(), equal_var=False)
print("Welch: INTL edge (%.2f) vs regular home HFA same seasons (%.2f): diff=%.2f, p=%.3f" % (x.mean(), ref.hfa_resid.mean(), x.mean()-ref.hfa_resid.mean(), tt.pvalue))
# OOS: fit-window intl edge applied to test window
fit_edge = neu[neu.intl & (neu.season<=2021)].hfa_resid.mean(); te = neu[neu.intl & (neu.season>=2022)].copy()
for nm, k in [("k=0 (spec neutral)",0.0), ("k=0.75 (spec intl mid)",0.75), ("k=fit-window intl edge",fit_edge), ("k=league 2.0",2.0), ("k=nfelo hfa_pts", te.hfa_pts.values)]:
    e = te.margin - te.rating_dif - k; print("  OOS 2022-25 intl (n=%d) %-24s MAE=%.2f bias=%+.2f" % (e.notna().sum(), nm, e.abs().mean(), e.mean()))
e = te.spread_err_mkt; print("  OOS market on same games: MAE=%.2f bias=%+.2f" % (e.abs().mean(), e.mean()))
print("\nGame list (INTL):")
print(neu[neu.intl][["season","week","away","home","venue","spread_line","result","rating_dif","hfa_resid","spread_err_mkt","hfa_pts"]].round(2).to_string(index=False))
print("\nDomestic neutral REG list:")
print(neu[~neu.intl & neu.game_type.eq("REG")][["season","week","away","home","stadium","spread_line","result","rating_dif","hfa_resid","spread_err_mkt","hfa_pts"]].round(2).to_string(index=False))
# nfelo behaviour: how often is hfa_pts nonzero at neutral sites?
print("\nnfelo hfa_pts at neutral REG sites: |hfa|>0.25 in %d of %d rated games; mean %.2f, mean abs %.2f" %
      ((neu[neu.game_type.eq("REG")].hfa_pts.abs()>0.25).sum(), neu[neu.game_type.eq("REG")].hfa_pts.notna().sum(),
       neu[neu.game_type.eq("REG")].hfa_pts.mean(), neu[neu.game_type.eq("REG")].hfa_pts.abs().mean()))
