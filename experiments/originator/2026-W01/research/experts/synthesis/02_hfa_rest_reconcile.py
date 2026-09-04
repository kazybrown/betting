"""Synthesis check 02: reconcile the HFA expert (constant 1.75, no tz, neutral override) with the
schedule expert (rest/bye once, hfa_mod composition, tz double count). Rating line = nfelo Elo dif +
nfelo QB mod + HFA variant, home perspective (pred_margin). Fit <=2021 shown as in-sample; 2022-25 OOS."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research")
import numpy as np, pandas as pd, statsmodels.api as sm
from kit import load_games, load_nfelo

g = load_games(); n = load_nfelo()
for c in ["home_time_advantage_mod", "div_game_mod", "dif_surface_mod", "home_bye_mod", "away_bye_mod", "hfa_base_mod", "home_net_qb_mod"]:
    n[c] = n[c].fillna(0)
comp = n.hfa_base_mod + n.home_time_advantage_mod + n.div_game_mod + n.dif_surface_mod + n.home_bye_mod + n.away_bye_mod
res = n.hfa_mod - comp
print(f"[A] hfa_mod - (hfa_base + tz + div + surface + home_bye + away_bye): mean {res.mean():.4f} sd {res.std():.4f} "
      f"share |res|<0.01 = {(res.abs()<0.01).mean():.3f}  (n={len(n)})")
res2 = n.nfelo_dif_base - ((n.starting_nfelo_home - n.starting_nfelo_away) + n.hfa_mod + n.home_net_qb_mod)
print(f"[A] nfelo_dif_base - (elo_dif + hfa_mod + qb): mean {res2.mean():.3f} sd {res2.std():.3f} share |res|<0.01 = {(res2.abs()<0.01).mean():.3f}")
print("[A] tz share of hfa_mod: corr(hfa_mod, tz) =", round(np.corrcoef(n.hfa_mod, n.home_time_advantage_mod)[0, 1], 3),
      "| as-built (hfa_mod + tz) carries tz at 2x by construction")
print("[A] hfa_base_mod/25 mean by era:", n.assign(era=pd.cut(n.season, [2008, 2013, 2019, 2021, 2025, 2026], labels=["09-13", "14-19", "20-21", "22-25", "26"]))
      .groupby("era", observed=True).hfa_base_mod.mean().div(25).round(2).to_dict())
print("[A] bye-mod sign check (Elo, home perspective): home_bye_mod mean when >0:", round(n.home_bye_mod[n.home_bye_mod > 0].mean(), 1),
      "| away_bye_mod mean when <0:", round(n.away_bye_mod[n.away_bye_mod < 0].mean(), 1))

m = g.merge(n[["gid", "starting_nfelo_home", "starting_nfelo_away", "hfa_mod", "hfa_base_mod", "home_time_advantage_mod", "div_game_mod",
               "dif_surface_mod", "home_bye_mod", "away_bye_mod", "home_net_qb_mod", "nfelo_dif_base"]], on="gid", how="inner")
m = m[(m.game_type == "REG") & m.mkt_spread.notna() & (~m.neutral) & (m.week >= 2)].copy()
m["core"] = (m.starting_nfelo_home - m.starting_nfelo_away + m.home_net_qb_mod) / 25.0   # rating dif + QB, pts, home persp.
m["rest_diff"] = (m.home_rest - m.away_rest).clip(-7, 7)
m["bye_side"] = np.where(m.home_rest >= 13, 1, np.where(m.away_rest >= 13, -1, 0))
m["bye_pts_nfelo"] = (m.home_bye_mod + m.away_bye_mod) / 25.0          # pts toward home in nfelo's line
print(f"[B] nfelo bye mod, pts to the bye team: home bye {m.bye_pts_nfelo[m.bye_side==1].mean():.2f} (n={int((m.bye_side==1).sum())}) | "
      f"away bye {(-m.bye_pts_nfelo[m.bye_side==-1]).mean():.2f} (n={int((m.bye_side==-1).sum())})")
as_built_site = (m.hfa_mod + m.home_time_advantage_mod) / 25.0
print(f"[B] as-built PFF/Cole site HFA (hfa_mod+tz)/25: mean {as_built_site.mean():.2f} sd {as_built_site.std():.2f} | "
      f"bye pts embedded for home-bye games {(m.home_bye_mod/25)[m.bye_side==1].mean():.2f}; blend carries 0.46*nfelo + 0.54*asbuilt + spec 0.75 = "
      f"{(0.46*m.bye_pts_nfelo[m.bye_side==1].mean() + 0.54*(m.home_bye_mod/25)[m.bye_side==1].mean() + 0.75):.2f} pts vs market ~1.0")

variants = {
    "const 1.25": 1.25, "const 1.5": 1.5, "const 1.75": 1.75, "const 2.0": 2.0, "const 2.5": 2.5,
    "hfa_base_mod/25": m.hfa_base_mod / 25, "hfa_mod/25 (incl bye,div,tz)": m.hfa_mod / 25,
    "(hfa_mod+tz)/25 AS-BUILT": as_built_site, "(hfa_base+tz)/25": (m.hfa_base_mod + m.home_time_advantage_mod) / 25,
    "const 1.75 + 0.15*rest_diff (RECONCILED)": 1.75 + 0.15 * m.rest_diff,
    "hfa_base/25 + 0.15*rest_diff": m.hfa_base_mod / 25 + 0.15 * m.rest_diff,
}
rng = np.random.default_rng(0)
def boot_diff(a, b, B=2000):
    d = np.abs(a) - np.abs(b); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for win, lab in [((2015, 2021), "IN-SAMPLE 2015-21"), ((2022, 2025), "OOS 2022-25")]:
    w = m[(m.season >= win[0]) & (m.season <= win[1])]
    ref = w.margin - (w.core + 1.75)
    print(f"[C] {lab} n={len(w)} | market MAE {np.abs(w.spread_err_mkt).mean():.3f}")
    for k, v in variants.items():
        hfa = v if np.isscalar(v) else v.loc[w.index]
        e = w.margin - (w.core + hfa)
        d, lo, hi = boot_diff(e.values, ref.values)
        bye = w.bye_side != 0
        bye_bias = (e[bye] * w.bye_side[bye]).mean()   # + = bye team beat the line
        print(f"    {k:42s} MAE {np.abs(e).mean():.3f}  bias {e.mean():+.2f}  dMAE vs 1.75 {d:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"bye-team bias {bye_bias:+.2f} (n={int(bye.sum())})")

# rest pricing: market and realized slopes on rest_diff, 2009-25
X = sm.add_constant(m[["rest_diff"]])
mk = sm.OLS(-m.mkt_spread - (m.core + 1.75), X).fit(cov_type="HC1")
rl = sm.OLS(m.margin - (m.core + 1.75), X).fit(cov_type="HC1")
print(f"[D] 2009-25 n={len(m)}: market prices rest_diff at {mk.params.rest_diff:+.3f}/day (se {mk.bse.rest_diff:.3f}); "
      f"realized {rl.params.rest_diff:+.3f}/day (se {rl.bse.rest_diff:.3f})")
byes = m[m.bye_side != 0]
print(f"[D] bye games n={len(byes)}: market prices bye team {((-byes.mkt_spread - (byes.core+1.75))*byes.bye_side).mean():+.2f}; "
      f"realized vs rating line {((byes.margin - (byes.core+1.75))*byes.bye_side).mean():+.2f} (se {((byes.margin - (byes.core+1.75))*byes.bye_side).std()/np.sqrt(len(byes)):.2f})")

# neutral-site tz leak
neu = g.merge(n[["gid", "home_time_advantage_mod", "hfa_base_mod", "hfa_mod"]], on="gid", how="inner")
neu = neu[neu.neutral]
print(f"[E] neutral-site games with nfelo rows n={len(neu)}: |tz mod|>2.5 Elo in {int((neu.home_time_advantage_mod.abs()>2.5).sum())}; "
      f"hfa_base_mod==0 in {int((neu.hfa_base_mod.fillna(0)==0).sum())}; mean |hfa_mod+tz|/25 = {((neu.hfa_mod+neu.home_time_advantage_mod.fillna(0)).abs()/25).mean():.2f} pts")
# West-at-1pm check is in schedule/04_travel.py; short-rest asymmetric count
sr = m[((m.home_rest <= 5) & (m.away_rest >= 6)) | ((m.away_rest <= 5) & (m.home_rest >= 6))]
print(f"[F] asymmetric short-rest games 2009-25 (one side <=5 days, other >=6): n={len(sr)}, since 2021: {int((sr.season>=2021).sum())}")
