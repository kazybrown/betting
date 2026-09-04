"""Theory 2 addendum: score the lines in historic_projected_spreads.csv (live nfelo v3 snapshots, 2021-2025)
against results. Its 'home_line_close' is a mid-week snapshot (it disagrees with the nflverse close in 53% of
games, e.g. 2021_09_GB_KC +1.0 vs -7.0), so the benchmark here is the nflverse closing line (mkt)."""
import numpy as np, pandas as pd
from common import load, paired_mae_test, D
m = load(verbose=False)
h = pd.read_csv(D / "historic_projected_spreads.csv", low_memory=False)
h = h[h.season <= 2025]
h["gid"] = h.game_id.str.replace("_LAR_", "_LA_").str.replace("_OAK_", "_LV_")
x = m.merge(h[["gid", "home_line_pre_regression", "home_closing_line_rounded_nfelo", "home_line_close", "market_regression_factor"]].rename(
    columns={"home_line_close": "mkt_snapshot"}), on="gid", how="inner")
print(f"joined {len(x)} games, seasons {x.season.min()}-{x.season.max()}")
print(f"historic pre-regression line vs nfelo_lin (backfilled series, 25 Elo/pt): mean |diff| = {np.abs(x.home_line_pre_regression - x.nfelo_lin).mean():.2f} pts, corr {np.corrcoef(x.home_line_pre_regression, x.nfelo_lin)[0,1]:.3f}")
print(f"historic 'close' snapshot vs nflverse close: share equal {(np.abs(x.mkt_snapshot - x.mkt) < 0.01).mean():.3f}, mean |diff| {np.abs(x.mkt_snapshot - x.mkt).mean():.2f}")
rows = []
for per, d in {"2021 (train)": x[x.season == 2021], "2022-25 (test)": x[x.test], "2021-25": x}.items():
    for lab, col in [("nflverse close (benchmark)", "mkt"), ("historic mkt snapshot", "mkt_snapshot"), ("historic pre-regression (unregressed)", "home_line_pre_regression"),
                     ("historic regressed (published)", "home_closing_line_rounded_nfelo"), ("nfelo_lin (backfill, 25/pt)", "nfelo_lin"), ("nfelo_close (backfill)", "nfelo_close")]:
        e = (d.margin + d[col]).values
        dm, lo, hi, p, n = paired_mae_test(e, d.err_mkt.values)
        rows.append(dict(period=per, line=lab, n=n, MAE=np.abs(e).mean(), dMAE_vs_close=dm, ci=f"[{lo:.3f},{hi:.3f}]", p=p))
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(3).to_string(index=False))
# implied w from the file's own regression: regressed = pre + f*(mkt - pre)  => engine weight = 1-f
print(f"\nnfelo's own engine weight (1 - market_regression_factor) by season: "
      f"{(1 - x.groupby('season').market_regression_factor.mean()).round(2).to_dict()}")
# Optimal w using the historic pre-regression line vs nflverse close, test period
te = x[x.test]
grid = np.arange(0, 1.01, 0.05); maes = [np.abs(te.margin + w*te.home_line_pre_regression + (1-w)*te.mkt).mean() for w in grid]
print(f"test-period (in-sample) optimal w for historic pre-regression line vs nflverse close: w={grid[int(np.argmin(maes))]:.2f}, MAE grid at 0/.1/.2/.3/.5/1: "
      f"{[round(maes[i],3) for i in (0,2,4,6,10,20)]}")
