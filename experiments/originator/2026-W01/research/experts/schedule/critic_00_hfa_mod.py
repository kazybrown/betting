"""CRITIC 00 - What is nfelo's hfa_mod? The expert claims (T4a/T6) 'hfa_mod already contains home_time_advantage_mod
(verified in 00b)', so ORIGINATOR's site HFA (hfa_mod + home_time_advantage_mod)/25 double-counts the tz term.
00b's output only showed that hfa_mod - hfa_base - tz has sd ~20 Elo, i.e. hfa_mod has a large extra component.
Here: regress hfa_mod on its candidate parts; cross-check with historic_projected_spreads (home_net_HFA_mod, base_hfa, time_mod)."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from kit import load_nfelo
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
n = load_nfelo().dropna(subset=["nfelo_dif_base"]).copy()
for c in ["hfa_base_mod", "home_time_advantage_mod", "div_game_mod", "dif_surface_mod", "home_bye_mod", "away_bye_mod", "home_net_qb_mod"]:
    n[c] = n[c].fillna(0)
n["extra"] = n.hfa_mod - n.hfa_base_mod - n.home_time_advantage_mod
print("A. hfa_mod summary by season (mean hfa_mod, hfa_base, tz, extra = hfa_mod - base - tz; sd extra)")
print(n.groupby("season").agg(hfa_mod=("hfa_mod", "mean"), base=("hfa_base_mod", "mean"), tz=("home_time_advantage_mod", "mean"), extra=("extra", "mean"), extra_sd=("extra", "std")).round(2).T)
print("\nB. OLS hfa_mod ~ hfa_base_mod + tz + div + surface + bye mods (does tz enter with coef ~1?)")
X = sm.add_constant(n[["hfa_base_mod", "home_time_advantage_mod", "div_game_mod", "dif_surface_mod", "home_bye_mod", "away_bye_mod"]])
r = sm.OLS(n.hfa_mod, X).fit()
print(pd.DataFrame({"coef": r.params, "se": r.bse}).round(3).T); print("  R2 %.3f  resid sd %.2f" % (r.rsquared, np.sqrt(r.mse_resid)))
r2 = sm.OLS(n.hfa_mod, sm.add_constant(n[["hfa_base_mod"]])).fit()
print("  hfa_mod ~ hfa_base only: coef %.3f R2 %.3f resid sd %.2f" % (r2.params.iloc[1], r2.rsquared, np.sqrt(r2.mse_resid)))
print("  corr(extra, tz) = %.3f | corr(hfa_mod - base, tz) = %.3f" % (np.corrcoef(n.extra, n.home_time_advantage_mod)[0, 1], np.corrcoef(n.hfa_mod - n.hfa_base_mod, n.home_time_advantage_mod)[0, 1]))
# is the 'extra' team-specific (a per-team home-field term)?
n["home"] = n.game_id.str.split("_").str[3]
byteam = n.groupby("home").extra.agg(["mean", "std", "count"]).round(1)
print("\nC. 'extra' (hfa_mod - base - tz) by home team (Elo): is it a team-specific HFA?")
print(byteam.sort_values("mean").T)
anova_between = n.groupby("home").extra.mean().var(); within = n.groupby("home").extra.var().mean()
print("  between-team variance of team means %.1f vs within-team variance %.1f" % (anova_between, within))
# within team over time
print("  extra by team-season sd (mean over teams of the within-season sd):", round(n.groupby(["home", "season"]).extra.std().mean(), 2))
# does nfelo_dif_base use hfa_mod or hfa_base + tz?
base = n.starting_nfelo_home - n.starting_nfelo_away + n.home_net_qb_mod + n.div_game_mod + n.dif_surface_mod + n.home_bye_mod + n.away_bye_mod
print("\nD. nfelo_dif_base - (elo dif + qb + div + surf + byes) compared with hfa_base + tz vs hfa_mod vs hfa_mod + tz")
rem = n.nfelo_dif_base - base
for lab, cand in [("hfa_base + tz", n.hfa_base_mod + n.home_time_advantage_mod), ("hfa_mod", n.hfa_mod), ("hfa_mod + tz", n.hfa_mod + n.home_time_advantage_mod), ("hfa_base", n.hfa_base_mod)]:
    d = rem - cand; print("  %-16s mean %+7.2f sd %6.2f  |d|>1 Elo rows: %d / %d" % (lab, d.mean(), d.std(), (d.abs() > 1).sum(), len(d)))
# cross-check with historic_projected_spreads (2021+): home_net_HFA_mod, base_hfa, time_mod
h = pd.read_csv("/home/user/originator-2026-w01/research/data/historic_projected_spreads.csv", low_memory=False)
h["gid"] = h.game_id
m = n.merge(h[["gid", "home_net_HFA_mod", "base_hfa", "time_mod", "home_net_bye_mod", "surface_mod", "div_mod"]].rename(columns={"home_net_bye_mod": "h_net_bye"}), on="gid", how="inner")
print("\nE. cross-check vs historic_projected_spreads (2021+ overlap n=%d)" % len(m))
for a, b in [("hfa_mod", "home_net_HFA_mod"), ("hfa_base_mod", "base_hfa"), ("home_time_advantage_mod", "time_mod"), ("dif_surface_mod", "surface_mod"), ("div_game_mod", "div_mod")]:
    d = (m[a].fillna(0) - m[b].fillna(0)); print("  %-24s vs %-18s mean diff %+7.3f sd %6.3f corr %.3f" % (a, b, d.mean(), d.std(), np.corrcoef(m[a].fillna(0), m[b].fillna(0))[0, 1]))
d2 = m.home_net_HFA_mod.fillna(0) - m.base_hfa.fillna(0) - m.time_mod.fillna(0)
print("  home_net_HFA_mod - base_hfa - time_mod: mean %+.3f sd %.3f (if ~0, net HFA = base + tz exactly)" % (d2.mean(), d2.std()))
print("  same in nfelo_games 2021+: hfa_mod - hfa_base - tz mean %+.3f sd %.3f" % (m.extra.mean(), m.extra.std()))
print("  sample rows:"); print(m[["gid", "hfa_mod", "hfa_base_mod", "home_time_advantage_mod", "home_net_HFA_mod", "base_hfa", "time_mod"]].head(6).to_string(index=False))
# what does the extra correlate with? home elo? season? dome? the 2020 near-zero
print("\nF. corr(extra, starting_nfelo_home) = %.3f ; corr(extra, elo dif) = %.3f" % (np.corrcoef(n.extra, n.starting_nfelo_home)[0, 1], np.corrcoef(n.extra, n.starting_nfelo_home - n.starting_nfelo_away)[0, 1]))
print("   extra: mean %.2f sd %.2f; quantiles:" % (n.extra.mean(), n.extra.std()), n.extra.quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_dict())
