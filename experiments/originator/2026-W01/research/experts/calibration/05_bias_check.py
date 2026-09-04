"""Side check (not one of the four theories): mean signed error (home bias) of the market and of nfelo_lin by period.
err = margin + line; positive = home teams beat the number."""
import numpy as np, pandas as pd
from scipy import stats
from common import load
m = load(verbose=False)
rows = []
for per, d in {"2009-13": m[m.season <= 2013], "2014-18": m[(m.season >= 2014) & (m.season <= 2018)], "2019-21": m[(m.season >= 2019) & (m.season <= 2021)],
               "2022-25": m[m.test], "2022-25 REG": m[m.test & ~m.post], "all": m}.items():
    r = {"period": per, "n": len(d)}
    for c in ("err_mkt", "err_nfelo_lin", "err_nfelo_noqb"):
        t = stats.ttest_1samp(d[c], 0)
        r[c] = d[c].mean(); r[c + "_p"] = t.pvalue
    r["mean_hfa_pts_nfelo"] = d.hfa_pts.mean(); r["mean_mkt_line"] = d.mkt.mean(); r["mean_margin"] = d.margin.mean()
    rows.append(r)
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nby season (market err mean, nfelo_lin err mean, nfelo hfa pts):")
print(m.groupby("season").agg(n=("margin", "size"), mkt_bias=("err_mkt", "mean"), nfelo_bias=("err_nfelo_lin", "mean"), hfa_pts=("hfa_pts", "mean"), margin=("margin", "mean")).round(2).to_string())
