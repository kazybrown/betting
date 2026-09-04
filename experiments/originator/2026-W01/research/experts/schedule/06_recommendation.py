"""06 - Combined OOS evaluation (2022-25) on an ORIGINATOR proxy of (a) no schedule adjustments, (b) the spec's section-5
schedule table as written, (c) candidate replacements. PFF/Cole have no history, so the proxy is
   proxy = 0.46 * nfelo_component + 0.54 * rating_only_line      (rating-only = nfelo ratings + base HFA + QB + div + surface, NO rest / tz mods,
                                                                   i.e. what the PFF/Cole spread paths look like)
with nfelo_component = nfelo raw (unregressed, incl. its bye + tz mods) [primary] or nfelo published close [secondary].
Reports MAE / RMSE on all test games, on affected games, signed bias on affected games, and ATS vs close."""
import sys; sys.path.insert(0, "/home/user/originator-2026-w01/research/experts/schedule")
import numpy as np, pandas as pd
from common import build, desc, ats_side
pd.set_option("display.width", 250)
m = build()
d = m.dropna(subset=["nfelo_dif_base"]).copy()
d["final"] = ((d.season >= 2021) & (d.week == 18)) | ((d.season <= 2020) & (d.week == 17))
d["rd"] = np.where(d.rest_valid, (d.home_rest - d.away_rest).clip(-7, 7), 0).astype(float)
d["bye_sgn"] = d.home_bye.astype(float) - d.away_bye.astype(float)
d["short_sgn"] = d.away_short.astype(float) - d.home_short.astype(float)        # + = home is the rested side
d["west_early_sgn"] = ((d.away_off <= -2) & d.early & ~d.neutral).astype(float) - ((d.home_off <= -2) & d.early & (d.away_off > -2) & ~d.neutral).astype(float)  # + = away is the west team at 1pm ET (home helped)
d["xc_sgn"] = -((d.tz_diff.abs() == 3) & ~d.neutral).astype(float)                # traveller (away) helped
d["nfelo_mods"] = d.nfelo_bye_pts + d.nfelo_tz_pts                                # pts toward home already inside nfelo raw

def evaluate(name, line, dd, affected=None):
    e = dd.margin + line
    out = "  %-58s MAE %.4f RMSE %.4f" % (name, e.abs().mean(), np.sqrt((e ** 2).mean()))
    if affected is not None and affected.sum() > 0:
        ea = e[affected]; w, l, p, pct, pv = ats_side(dd.err_mkt[affected], np.sign(dd.mkt_spread[affected] - line[affected]))
        out += " | affected n=%4d MAE %.3f bias %+.3f | ATS vs close %d-%d %.3f" % (affected.sum(), ea.abs().mean(), ea.mean(), w, l, pct)
    print(out); return e

for comp_name, comp in [("nfelo RAW (unregressed)", "nraw_line"), ("nfelo published CLOSE (market-regressed)", "nfelo_home_line_close")]:
    t = d[d.test].copy()
    base = 0.46 * t[comp] + 0.54 * t.rate_line
    print("\n=== nfelo component = %s | test 2022-25 REG n=%d ===" % (comp_name, len(t)))
    print("  market close MAE %.4f RMSE %.4f | nfelo component alone MAE %.4f | rating-only alone MAE %.4f" % (t.err_mkt.abs().mean(), np.sqrt((t.err_mkt ** 2).mean()), (t.margin + t[comp]).abs().mean(), t.err_rate.abs().mean()))
    evaluate("(a) proxy, no schedule adjustments", base, t)
    # (b) spec as written, midpoints: short -0.9 (only when opponent not short), bye +0.75, west-to-east 1pm kick -0.6. ORIGINATOR sign: negative = home favored
    spec = base - 0.9 * t.short_sgn - 0.75 * t.bye_sgn - 0.6 * t.west_early_sgn
    aff = (t.short_sgn != 0) | (t.bye_sgn != 0) | (t.west_early_sgn != 0)
    evaluate("(b) + spec table (short -0.9, bye +0.75, west@1pm -0.6)", spec, t, aff)
    evaluate("(b1) spec bye only (+0.75 on top of nfelo's mod)", base - 0.75 * t.bye_sgn, t, t.bye_sgn != 0)
    evaluate("(b2) spec west@1pm only (-0.6 to west team)", base - 0.6 * t.west_early_sgn, t, t.west_early_sgn != 0)
    # (c) candidates
    stripped = 0.46 * (t[comp] + t.nfelo_mods) + 0.54 * t.rate_line      # remove nfelo's bye + tz mods from the nfelo component
    evaluate("(c0) strip nfelo bye+tz mods, no adjustment", stripped, t, t.nfelo_mods != 0)
    evaluate("(c1) strip mods, + rest 0.15/day (cap +-7) whole blend", stripped - 0.15 * t.rd, t, t.rd != 0)
    evaluate("(c2) strip mods, + bye 1.0 only (whole blend)", stripped - 1.0 * t.bye_sgn, t, t.bye_sgn != 0)
    evaluate("(c3) keep nfelo mods, + 0.54*0.15/day to non-nfelo share", base - 0.54 * 0.15 * t.rd, t, t.rd != 0)
    evaluate("(c4) strip tz only, keep nfelo bye, + 0.54*1.0 bye", 0.46 * (t[comp] + t.nfelo_tz_pts) + 0.54 * t.rate_line - 0.54 * t.bye_sgn, t, t.bye_sgn != 0)
    evaluate("(c5) (c1) + exploratory: +1.0 to 3-zone traveller", stripped - 0.15 * t.rd - 1.0 * t.xc_sgn, t, t.xc_sgn != 0)
    # final-week handling: shrink to market
    for a in [1.0, 0.5, 0.25, 0.0]:
        line = np.where(t.final, a * (stripped - 0.15 * t.rd) + (1 - a) * t.mkt_spread, stripped - 0.15 * t.rd)
        evaluate("(d) (c1) + final week: engine weight a=%.2f vs market" % a, pd.Series(line, index=t.index), t, t.final)
    # bias check on bye games for the candidates (should be ~0 when right-sized)
    b = t[t.bye_sgn != 0]
    for nm, line in [("(a)", base), ("(b1)", base - 0.75 * t.bye_sgn), ("(c1)", stripped - 0.15 * t.rd), ("(c2)", stripped - 1.0 * t.bye_sgn), ("market", t.mkt_spread)]:
        e = (b.margin + line[b.index]) * b.bye_sgn; print("      bye games n=%d: %-7s bye-team bias %+.2f (se %.2f)" % (len(b), nm, e.mean(), e.std() / np.sqrt(len(e))))

# rolling-origin check for the rest term on the rating-only line: fit k per year on all prior seasons, test that season
print("\n=== rolling-origin (fit on all seasons < s, test s) of k*rest_diff on the rating-only line, seasons 2016-2025 ===")
rows = []
for s in range(2016, 2026):
    f = d[(d.season < s) & d.rest_valid]; t = d[(d.season == s) & d.rest_valid]
    ks = np.arange(0, 0.51, 0.025); k = ks[int(np.argmin([(f.margin + f.rate_line - kk * f.rd).abs().mean() for kk in ks]))]
    e0 = t.margin + t.rate_line; e1 = e0 - k * t.rd; aff = t.rd != 0
    rows.append((s, k, e0.abs().mean(), e1.abs().mean(), e0[aff].abs().mean(), e1[aff].abs().mean(), (e0[aff] * np.sign(t.rd[aff])).mean(), (e1[aff] * np.sign(t.rd[aff])).mean(), aff.sum()))
r = pd.DataFrame(rows, columns=["season", "k_fit", "MAE_base", "MAE_adj", "MAE_base_aff", "MAE_adj_aff", "bias_rested_base", "bias_rested_adj", "n_aff"])
print(r.round(3).to_string(index=False))
print("  pooled affected games: bias of rested side base %+.3f -> adj %+.3f; MAE base %.4f -> adj %.4f (n=%d)" %
      ((r.bias_rested_base * r.n_aff).sum() / r.n_aff.sum(), (r.bias_rested_adj * r.n_aff).sum() / r.n_aff.sum(), (r.MAE_base_aff * r.n_aff).sum() / r.n_aff.sum(), (r.MAE_adj_aff * r.n_aff).sum() / r.n_aff.sum(), r.n_aff.sum()))
