"""critic_06: concrete sizing of the revised T4 rule under the leak-free season-to-date league
reference: total_adj = clip(b * epa_sum_std, -cap, +cap) for b in {3,4,5,6}, cap in {2,3}.
Scored (i) fit-free on the 2009-19-fitted BASE, OOS 2023-25, and (ii) pooled rolling-origin over the
10 available test seasons (2013-19, 2023-25). Also the mean adjustment size per rule.
"""
import numpy as np, pandas as pd
from critic_common import *  # noqa

m = load_games_table(); m = add_std_centered(m)
d = combos(m, "r8d"); d["epa_sum_s"] = d.h_epa_off_r8s + d.a_epa_off_r8s + d.h_epa_def_r8s + d.a_epa_def_r8s
d = reg_sample(d, BASE + ["lg_blend", "epa_sum", "epa_sum_s"])
tr, te = d[d.train], d[d.test]
fb, pb = fit_pred(tr, te, BASE); res_b = te.total_pts.values - pb
print(f"epa_sum_s: train SD={tr.epa_sum_s.std():.3f}, test SD={te.epa_sum_s.std():.3f}; train coef (BASE + epa_sum_s) = {fit_pred(tr, te, BASE + ['epa_sum_s'])[0].params['epa_sum_s']:+.2f}")
print(f"{'rule':28s} {'OOS 2023-25 dMAE [CI]':>28s} {'dRMSE':>7s} {'mean|adj|':>9s} {'O/U':>16s} | {'pooled rolling-origin dMAE [CI] (n=2381)':>40s}  seasons better")
for b in (3.0, 4.0, 5.0, 6.0):
    for cap in (2.0, 3.0):
        adj = np.clip(b * te.epa_sum_s.values, -cap, cap)
        dm, lo, hi, n = paired_mae_ci(te.total_pts - (pb + adj), res_b)
        w, l, pu = ou_rate(pb + adj, te.mkt_total, te.total_pts)
        e0, e1, better = [], [], 0
        for Y in list(range(2013, 2020)) + [2023, 2024, 2025]:
            a, bb = d[d.season < Y], d[d.season == Y]
            f0, p0 = fit_pred(a, bb, BASE); a1 = np.clip(b * bb.epa_sum_s.values, -cap, cap)
            e0 += list(bb.total_pts.values - p0); e1 += list(bb.total_pts.values - (p0 + a1))
            better += int(mae(p0 + a1, bb.total_pts) < mae(p0, bb.total_pts))
        dm2, lo2, hi2, n2 = paired_mae_ci(e1, e0)
        print(f"clip({b:.0f}*epa_sum_s, +/-{cap:.0f}){'':9s} {ci_str(dm, lo, hi):>28s} {rmse(pb+adj, te.total_pts)-rmse(pb, te.total_pts):+7.3f} {np.abs(adj).mean():9.2f} {w}-{l}-{pu} ({w/(w+l):.3f}) | {ci_str(dm2, lo2, hi2):>40s}  {better}/10")
print(f"expert's own rule for reference: clip(4*epa_sum [prior-season ref], +/-3): OOS {ci_str(*paired_mae_ci(te.total_pts - (pb + np.clip(4*te.epa_sum.values, -3, 3)), res_b)[:3])}")
