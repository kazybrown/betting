"""Synthesis check 05: reproduce the QB stint table on the QB expert's prospective D3 events
(qb_games_defs.csv, built by experts/qb/01,02,02b) restricted to in-season games where only one team
has a downgrade stint; fit window <=2021 vs OOS 2022-25. Also the nfelo-embedded adjustment (A_nfelo)
by stint for the net-of-embedded rule. Signs: + = penalty on the backup team."""
import numpy as np, pandas as pd, statsmodels.api as sm
m = pd.read_csv("/home/user/originator-2026-w01/research/experts/qb/qb_games_defs.csv", low_memory=False)
print("late_rest_wk values:", m.late_rest_wk.value_counts(dropna=False).to_dict())
late = m.late_rest_wk.astype(str).isin(["1", "1.0", "True"])
one = (m.net_D3 != 0) & ~((m.home_down == 1) & (m.away_down == 1))
stint = np.where(m.home_down == 1, m.home_stint3, np.where(m.away_down == 1, m.away_stint3, np.nan))
m["stint_bin"] = pd.cut(pd.Series(stint), [0, 1, 3, 99], labels=["1st", "2nd-3rd", "4th+"])
sgn = m.net_D3
def table(d, lab):
    print(f"[{lab}]")
    for b in ["1st", "2nd-3rd", "4th+"]:
        e = d[one.loc[d.index] & (d.stint_bin == b)]
        s = sgn.loc[e.index]
        mk = -(e.mkt_minus_noqb * s); rl = -(e.resid_noqb * s); nf = -(e.qb_adj_pts * s); r538 = -(e.resid_base * s); rmk = -(e.resid_mkt * s)
        # cross-check the sign convention on the market column: penalty must be positive at 1st start
        print(f"   stint {b:7s} n={len(e):4d} market-implied {mk.mean():+.2f}+/-{mk.std()/np.sqrt(len(e)):.2f} | realized vs noQB {rl.mean():+.2f}+/-{rl.std()/np.sqrt(len(e)):.2f} "
              f"(median {rl.median():+.2f}) | nfelo embedded A {nf.mean():+.2f}+/-{nf.std()/np.sqrt(len(e)):.2f} (median {nf.median():+.2f}) | resid after nfelo {r538.mean():+.2f}+/-{r538.std()/np.sqrt(len(e)):.2f} | resid vs market {rmk.mean():+.2f}+/-{rmk.std()/np.sqrt(len(e)):.2f}")
ins = m[~late]
table(ins[ins.season <= 2021], "IN-SEASON, one-side downgrade, FIT 2009-21")
table(ins[ins.season >= 2022], "IN-SEASON, one-side downgrade, OOS 2022-25")
table(ins, "IN-SEASON, one-side downgrade, ALL 2009-25")
table(m[late], "LATE/REST-WEEK games (all seasons)")
# net-of-embedded arithmetic at the fit-window means
e = ins[(ins.season <= 2021) & one.loc[ins.index] & (ins.stint_bin == "1st")]
P = 2.5; A = (-(e.qb_adj_pts * sgn.loc[e.index])).mean()
print(f"[rule] P_stint1=2.5 ; mean A_nfelo at 1st start (fit) = {A:.2f} -> nfelo-share top-up 0.46*max(0,P-A) = {0.46*max(0,P-A):.2f}; "
      f"spec's -3.0 on top of a fully QB-adjusted blend would total {A+3.0:.2f} vs market-implied {(-(e.mkt_minus_noqb*sgn.loc[e.index])).mean():.2f}")
