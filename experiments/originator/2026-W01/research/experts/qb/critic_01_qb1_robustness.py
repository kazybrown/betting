"""CRITIC 01 (theory qb-1): attack the backup-QB stint table.
(A) data integrity: QB-name variants in games.csv fabricate 'QB changes'; rebuild D3 with qb_id and compare.
(B) robust estimators (median / Huber / 10% trimmed) for the stint-1 in-season realized & market-implied penalty, fit vs test.
(C) leave-one-season-out jackknife of the stint-1 coefficients.
(D) rolling-origin OOS (fit all prior seasons, test one) of the stint table on the no-QB line vs nfelo's own 538 adj.
(E) permutation placebo (flags shuffled within season) -> null distribution of the realized coefficient.
(F) 'week-before' placebo: the displaced starter's LAST game before the change.
(G) test of the untested sub-rule '3.5 when the displaced starter's 538 adj >= +3, 2.5 when <= +1'.
(H) fit-window-only stint table (what the parameter should be sized on).
Reads qb_games_defs.csv (built by 01/02/02b) + games_1999_2025.csv. Re-runnable."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
sys.path.insert(0, "/home/user/originator-2026-w01/research")
from kit import load_games, norm, mae
HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "qb_games_defs.csv", low_memory=False)
m = m[m.line_nfelo_noqb.notna() & m.mkt_spread.notna()].copy()
m["late"] = ((m.game_type != "REG") | (m.week >= 17)).astype(int)
print("sign check corr(mkt_spread, margin) =", round(np.corrcoef(m.mkt_spread, m.margin)[0, 1], 3), "(must be strongly negative)")

def coef(y, x, d, cov="HC1"):
    d = d.dropna(subset=[y, x]); r = sm.OLS(d[y], sm.add_constant(d[[x]])).fit(cov_type=cov)
    return r.params[x], r.bse[x], r.pvalues[x], int((d[x] != 0).sum())

# ---------------------------------------------------------------- (A) name variants
print("\n(A) QB-name integrity in games.csv (the expert keys career starts + changes on qb_name)")
g_all = load_games(1999); g_all["gdate"] = pd.to_datetime(g_all.gameday)
rows = []
for side in ["home", "away"]:
    t = g_all[["gid", "season", "week", "game_type", "gdate", f"{side}_team", f"{side}_qb_name", f"{side}_qb_id"]].copy()
    t.columns = ["gid", "season", "week", "game_type", "gdate", "team", "qb", "qb_id"]; t["side"] = side; rows.append(t)
tg = pd.concat(rows).sort_values(["gdate", "gid"]).reset_index(drop=True)
tg["team"] = tg.team.map(norm)
idn = tg.groupby("qb_id").qb.nunique(); multi = idn[idn > 1].index
print("  ids with >1 name spelling:", len(multi))
for i in multi:
    print("    ", i, tg[tg.qb_id == i].groupby("qb").size().to_dict())
nm = tg.groupby("qb").qb_id.nunique(); print("  names with >1 id:", nm[nm > 1].to_dict())

def build_d3(tg, key):
    """expert's 02b logic, keyed on `key` (qb name or qb_id)."""
    t = tg.copy(); t["k"] = t[key]
    t["career"] = t.groupby("k").cumcount()
    t = t.sort_values(["team", "gdate"]).reset_index(drop=True)
    t["prev_career"] = t.groupby("team").career.shift(1)
    down = []; st = []
    for team, grp in t.groupby("team", sort=False):
        cur = 0; last = None; c = 0
        for _, r in grp.iterrows():
            if r.week == 1 and r.game_type == "REG":
                cur = 0; c = 1
            elif r.k != last:
                c = 1; cur = 0 if last is None else int(r.career < r.prev_career)
            else:
                c += 1
            down.append(cur); st.append(c); last = r.k
    t["down"] = down; t["stint3"] = st
    return t

t_name = build_d3(tg, "qb"); t_id = build_d3(tg, "qb_id")
cmp = t_name[["gid", "side", "team", "season", "qb", "down", "stint3"]].merge(
    t_id[["gid", "side", "down", "stint3"]], on=["gid", "side"], suffixes=("_name", "_id"))
cmp = cmp[cmp.season >= 2009]
diff = cmp[(cmp.down_name != cmp.down_id) | ((cmp.down_name == 1) & (cmp.stint3_name != cmp.stint3_id))]
print("  team-games 2009+ where name-keyed and id-keyed D3 flags disagree:", len(diff))
print(diff.groupby("qb").size().sort_values(ascending=False).head(12).to_dict())
s1n = cmp[(cmp.down_name == 1) & (cmp.stint3_name == 1)]; s1i = cmp[(cmp.down_id == 1) & (cmp.stint3_id == 1)]
print(f"  stint-1 downgrade team-games: name-keyed={len(s1n)}  id-keyed={len(s1i)}  name-only={len(set(s1n.gid+s1n.side)-set(s1i.gid+s1i.side))}  id-only={len(set(s1i.gid+s1i.side)-set(s1n.gid+s1n.side))}")
print("  NOTE: Kyle Allen and Cam Newton share id 00-0034577 in nflverse (data error) -> id-keyed misses their CAR swaps; name-keyed is right there.")
# merge id-based flags to game level and rerun headline stint-1 regression
h = t_id[t_id.side == "home"].set_index("gid"); a = t_id[t_id.side == "away"].set_index("gid")
gl = pd.concat([h[["down", "stint3"]].add_prefix("homeID_"), a[["down", "stint3"]].add_prefix("awayID_")], axis=1).reset_index()
m = m.merge(gl, on="gid", how="left")
for tag, hd, hs, ad, as_ in [("name", "home_down", "home_stint3", "away_down", "away_stint3"), ("id", "homeID_down", "homeID_stint3", "awayID_down", "awayID_stint3")]:
    both = (m[hd] == 1) & (m[ad] == 1)
    m[f"s1_{tag}"] = ((m[hd] == 1) & (m[hs] == 1)).astype(int) - ((m[ad] == 1) & (m[as_] == 1)).astype(int)
    m[f"net_{tag}"] = m[hd] - m[ad]
    d = m[~both & (m.late == 0) & ((m[f"s1_{tag}"] != 0) | (m[f"net_{tag}"] == 0))]
    out = []
    for y, nmn, sg in [("mkt_minus_noqb", "market", 1), ("resid_noqb", "realized", -1), ("qb_adj_pts", "538adj", -1), ("resid_base", "after538", -1), ("resid_mkt", "vs_mkt", -1)]:
        b, se, p, n = coef(y, f"s1_{tag}", d); out.append(f"{nmn}={sg*b:+.2f}±{se:.2f}")
    print(f"  headline stint-1 in-season clean, keyed on {tag:<4s}: n_event={n}  " + "  ".join(out))

# ---------------------------------------------------------------- (B) robust estimators
print("\n(B) stint-1 in-season clean sample: event-signed penalties, OLS vs robust (name-keyed, expert's sample)")
both = (m.home_down == 1) & (m.away_down == 1)
clean = m[~both & (m.late == 0) & ((m.s1_name != 0) | (m.net_name == 0))].copy()
ev = clean[clean.s1_name != 0].copy()
ev["pen_real"] = -(ev.resid_noqb * ev.s1_name); ev["pen_mkt"] = ev.mkt_minus_noqb * ev.s1_name; ev["pen_after538"] = -(ev.resid_base * ev.s1_name); ev["pen_vsmkt"] = -(ev.resid_mkt * ev.s1_name)
ctrl = clean[clean.s1_name == 0]
def robust_block(lab, e, c):
    for col in ["pen_mkt", "pen_real", "pen_after538", "pen_vsmkt"]:
        base_col = {"pen_mkt": "mkt_minus_noqb", "pen_real": "resid_noqb", "pen_after538": "resid_base", "pen_vsmkt": "resid_mkt"}[col]
        # control-group mean of the signed base variable is ~0 (intercept); report event stats net of control mean
        cm = c[base_col].mean()
        x = e[col].values; sgn = -1 if col != "pen_mkt" else 1
        mean = x.mean() - sgn * cm
        med = np.median(x) - sgn * np.median(c[base_col])
        tr = stats.trim_mean(x, 0.1) - sgn * stats.trim_mean(c[base_col], 0.1)
        # quantile regression on the full sample
        d = pd.concat([e, c]); qr = sm.QuantReg(d[base_col], sm.add_constant(d[["s1_name"]])).fit(q=0.5)
        rlm = sm.RLM(d[base_col], sm.add_constant(d[["s1_name"]]), M=sm.robust.norms.HuberT()).fit()
        print(f"  {lab:<22s} {col:<13s} n={len(x):3d} mean={mean:+.2f} (se {x.std(ddof=1)/np.sqrt(len(x)):.2f})  median={med:+.2f}  trim10={tr:+.2f}  QuantReg(q=.5)={sgn*qr.params['s1_name']:+.2f}±{qr.bse['s1_name']:.2f}  Huber={sgn*rlm.params['s1_name']:+.2f}±{rlm.bse['s1_name']:.2f}")
robust_block("ALL 2009-2025", ev, ctrl)
robust_block("FIT 2009-2021", ev[ev.season <= 2021], ctrl[ctrl.season <= 2021])
robust_block("TEST 2022-2025", ev[ev.season >= 2022], ctrl[ctrl.season >= 2022])
print("  2022-25 stint-1 realized penalties, sorted (pts):", np.sort(ev[ev.season >= 2022].pen_real.values).round(0).astype(int).tolist())

# ---------------------------------------------------------------- (C) jackknife
print("\n(C) leave-one-season-out jackknife, stint-1 in-season clean (name-keyed): market-implied / realized")
jk = []
for yr in sorted(clean.season.unique()):
    d = clean[clean.season != yr]
    bm, _, _, _ = coef("mkt_minus_noqb", "s1_name", d); br, _, _, n = coef("resid_noqb", "s1_name", d)
    jk.append((yr, bm, -br))
jk = pd.DataFrame(jk, columns=["dropped", "market", "realized"])
print("  market range [%.2f, %.2f]  realized range [%.2f, %.2f]" % (jk.market.min(), jk.market.max(), jk.realized.min(), jk.realized.max()))
print("  season-by-season event means (n, market, realized):")
ss = ev.groupby("season").agg(n=("pen_real", "size"), market=("pen_mkt", "mean"), realized=("pen_real", "mean")).round(2)
print(ss.T.to_string())

# ---------------------------------------------------------------- (D) rolling-origin OOS for the stint table
print("\n(D) rolling-origin OOS: stint table fitted on all prior in-season seasons, applied to season t on the no-QB line; vs nfelo's 538 adj")
ins = m[(m.late == 0) & m.nfelo_home_line_close.notna()].copy()
def ns(df, lo, hi, hd="home_down", hs="home_stint3", ad="away_down", as_="away_stint3"):
    return ((df[hd] == 1) & df[hs].between(lo, hi)).astype(int) - ((df[ad] == 1) & df[as_].between(lo, hi)).astype(int)
ins["s1"] = ns(ins, 1, 1); ins["s23"] = ns(ins, 2, 3); ins["s4"] = ns(ins, 4, 99)
tot = {"base": [], "tab": [], "s1only": [], "fixed3": [], "n538": [], "resc": []}; evtot = {"base": [], "tab": [], "n538": []}
for yr in range(2016, 2026):
    tr = ins[ins.season < yr]; te = ins[ins.season == yr]
    r = sm.OLS(tr.resid_noqb, sm.add_constant(tr[["s1", "s23", "s4"]])).fit(); pen = -r.params
    sc = sm.OLS(tr.resid_noqb, sm.add_constant(tr[["qb_adj_pts"]])).fit().params["qb_adj_pts"]
    e0 = te.margin + te.line_nfelo_noqb
    e_tab = e0 + pen.s1 * te.s1 + pen.s23 * te.s23 + pen.s4 * te.s4
    e_s1 = e0 + pen.s1 * te.s1
    e_fix = e0 + 3.0 * te.s1 + 1.75 * te.s23 + 0.5 * te.s4       # the expert's recommended table
    e_538 = te.margin + te.line_nfelo_base
    e_resc = e0 - sc * te.qb_adj_pts
    for k, e in [("base", e0), ("tab", e_tab), ("s1only", e_s1), ("fixed3", e_fix), ("n538", e_538), ("resc", e_resc)]:
        tot[k].append(np.abs(e).mean())
    evm = te.net_D3 != 0
    for k, e in [("base", e0), ("tab", e_tab), ("n538", e_538)]:
        evtot[k].append(np.abs(e[evm]).mean())
    print(f"  {yr}: fitted table {pen.s1:.2f}/{pen.s23:.2f}/{pen.s4:.2f} s538={sc:.2f} | MAE noQB={tot['base'][-1]:.3f} +table={tot['tab'][-1]:.3f} +s1only={tot['s1only'][-1]:.3f} +rec(3/1.75/.5)={tot['fixed3'][-1]:.3f} nfelo538={tot['n538'][-1]:.3f} 538x{sc:.2f}={tot['resc'][-1]:.3f} | event-only noQB={evtot['base'][-1]:.2f} table={evtot['tab'][-1]:.2f} 538={evtot['n538'][-1]:.2f} (ev n={int(evm.sum())})")
print("  seasons (of 10) where +table beats noQB: %d ; where nfelo538 beats +table: %d ; where rec table beats noQB: %d" % (
    sum(t < b for t, b in zip(tot['tab'], tot['base'])), sum(n < t for n, t in zip(tot['n538'], tot['tab'])), sum(f < b for f, b in zip(tot['fixed3'], tot['base']))))
print("  mean MAE 2016-25: noQB=%.4f +table=%.4f +s1only=%.4f +rec=%.4f nfelo538=%.4f 538-rescaled=%.4f" % tuple(np.mean(tot[k]) for k in ["base", "tab", "s1only", "fixed3", "n538", "resc"]))

# ---------------------------------------------------------------- (E) permutation placebo
print("\n(E) permutation placebo: shuffle the stint-1 flag across games within season (same count), 2000 draws, in-season clean sample")
rng = np.random.default_rng(3)
obs = -coef("resid_noqb", "s1_name", clean)[0]
null = []
x = clean.s1_name.values; seasons = clean.season.values
for _ in range(2000):
    xs = x.copy()
    for yr in np.unique(seasons):
        idx = np.where(seasons == yr)[0]; xs[idx] = rng.permutation(xs[idx])
    d = clean.assign(xp=xs); null.append(-sm.OLS(d.resid_noqb, sm.add_constant(d[["xp"]])).fit().params["xp"])
null = np.array(null)
print(f"  observed realized penalty {obs:+.2f}; placebo null mean {null.mean():+.2f} sd {null.std():.2f}; p(perm) = {(np.abs(null) >= abs(obs)).mean():.4f}")

# ---------------------------------------------------------------- (F) week-before placebo
print("\n(F) 'week-before' placebo: the displaced starter's LAST game before a stint-1 downgrade (should show ~0 market move if the measure is clean)")
tn = t_name[t_name.season >= 2009].sort_values(["team", "gdate"]).copy()
tn["next_down"] = tn.groupby("team").down.shift(-1); tn["next_stint"] = tn.groupby("team").stint3.shift(-1)
tn["next_week"] = tn.groupby("team").week.shift(-1); tn["next_season"] = tn.groupby("team").season.shift(-1)
tn["pre"] = ((tn.next_down == 1) & (tn.next_stint == 1) & (tn.next_season == tn.season)).astype(int)
hp = tn[tn.side == "home"].set_index("gid").pre; ap = tn[tn.side == "away"].set_index("gid").pre
m["pre_net"] = m.gid.map(hp).fillna(0).astype(int) - m.gid.map(ap).fillna(0).astype(int)
d = m[(m.late == 0) & (m.net_name == 0)]   # no backup on either side in the placebo game itself
for y, lab, sg in [("mkt_minus_noqb", "market move vs no-QB line", 1), ("resid_noqb", "realized vs no-QB line", -1), ("resid_mkt", "realized vs market", -1), ("qb_adj_pts", "538 adj", -1)]:
    b, se, p, n = coef(y, "pre_net", d); print(f"  {lab:<28s} coef(as penalty)={sg*b:+.2f}±{se:.2f} p={p:.3f} n_event={n}")

# ---------------------------------------------------------------- (G) sub-rule: displaced starter's 538 adj
print("\n(G) sub-rule test: split stint-1 in-season events by the DISPLACED starter's 538 adj (team's qb adj in its previous game, pts)")
tn["prev_adj"] = np.nan
adjmap = {}
for side in ["home", "away"]:
    s = m.set_index("gid")[f"{side}_538_qb_adj"] / 25.0
    adjmap[side] = s
tn["own_adj"] = [adjmap[s].get(g, np.nan) for s, g in zip(tn.side, tn.gid)]
tn["prev_adj"] = tn.groupby("team").own_adj.shift(1)
tn["prev_season2"] = tn.groupby("team").season.shift(1)
pa_h = tn[tn.side == "home"].set_index("gid").prev_adj; pa_a = tn[tn.side == "away"].set_index("gid").prev_adj
ev = ev.copy()
ev["prev_adj"] = np.where(ev.s1_name == 1, ev.gid.map(pa_h), ev.gid.map(pa_a))
ev["own_adj_now"] = np.where(ev.s1_name == 1, ev.home_538_qb_adj / 25, ev.away_538_qb_adj / 25)
ev["drop538"] = ev.prev_adj - ev.own_adj_now
print("  displaced starter's prior-game 538 adj (pts): describe:", ev.prev_adj.describe().round(2).to_dict())
for lab, mask in [("displaced adj >= +3", ev.prev_adj >= 3), ("+1 < adj < +3", (ev.prev_adj > 1) & (ev.prev_adj < 3)), ("adj <= +1", ev.prev_adj <= 1), ("adj <= -1", ev.prev_adj <= -1)]:
    e = ev[mask]
    if len(e) < 5: print(f"  {lab}: n={len(e)} too few"); continue
    print(f"  {lab:<22s} n={len(e):3d}  market={e.pen_mkt.mean():+.2f}±{e.pen_mkt.std()/np.sqrt(len(e)):.2f}  realized={e.pen_real.mean():+.2f}±{e.pen_real.std()/np.sqrt(len(e)):.2f}  538 drop={e.drop538.mean():+.2f}  after538={e.pen_after538.mean():+.2f}±{e.pen_after538.std()/np.sqrt(len(e)):.2f}")
r = sm.OLS(ev.pen_mkt, sm.add_constant(ev[["prev_adj"]].fillna(0))).fit(cov_type="HC1")
r2 = sm.OLS(ev.pen_real, sm.add_constant(ev[["prev_adj"]].fillna(0))).fit(cov_type="HC1")
print(f"  market penalty ~ displaced adj: slope {r.params.prev_adj:+.2f}±{r.bse.prev_adj:.2f} (p={r.pvalues.prev_adj:.3f}) | realized ~ displaced adj: slope {r2.params.prev_adj:+.2f}±{r2.bse.prev_adj:.2f} (p={r2.pvalues.prev_adj:.3f})")
r3 = sm.OLS(ev.pen_mkt, sm.add_constant(ev[["drop538"]].fillna(0))).fit(cov_type="HC1")
print(f"  market penalty ~ 538 DROP (prev adj - new adj): slope {r3.params.drop538:+.2f}±{r3.bse.drop538:.2f}, intercept {r3.params.const:+.2f}  (a continuous QB-gap input explains market pricing better than a flat value)")

# ---------------------------------------------------------------- (H) fit-window-only stint table on the clean in-season sample
print("\n(H) FIT-WINDOW-ONLY (2009-2021) stint table, in-season, opponent not a backup — the numbers the parameter should be sized on")
fit = m[(m.season <= 2021) & (m.late == 0) & ~both].copy()
fit["s1"] = ns(fit, 1, 1); fit["s23"] = ns(fit, 2, 3); fit["s4"] = ns(fit, 4, 99)
for y, lab, sg in [("mkt_minus_noqb", "market-implied", 1), ("resid_noqb", "realized", -1), ("qb_adj_pts", "nfelo 538 embeds", -1), ("resid_base", "after 538", -1)]:
    r = sm.OLS(fit[y], sm.add_constant(fit[["s1", "s23", "s4"]])).fit(cov_type="HC1")
    print(f"  {lab:<18s} 1st={sg*r.params.s1:+.2f}±{r.bse.s1:.2f}  2nd-3rd={sg*r.params.s23:+.2f}±{r.bse.s23:.2f}  4th+={sg*r.params.s4:+.2f}±{r.bse.s4:.2f}   (n_event {int((fit.s1!=0).sum())}/{int((fit.s23!=0).sum())}/{int((fit.s4!=0).sum())})")
