#!/usr/bin/env python3
"""Research-calibrated configuration of the ORIGINATOR engine (candidate v6).

Reuses the spec engine's loaders and audited adjustments, then applies the
parameter changes recommended by the expert panel via research_config.json.
The spec-default pipeline (originator_engine.py) is untouched; this module
writes final_research.json and a side-by-side comparison so both cards can
be inspected before anything is adopted.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
from originator_engine import (RAW, load_slate, load_nfelo, load_pff_ratings, load_cole_ratings,  # noqa: E402
                               round_half, norm_team)

CFG = json.loads((RUN / "research_config.json").read_text())
FEAT = {(r["away"], r["home"]): r for r in csv.DictReader(open(RUN / "research" / "week1_features.csv"))}


def env_adj(roof, wind_mph=None):
    """Environment table (totals): dome/closed/retractable = +dome; outdoor =
    base at league-average wind, linear in wind when a verified wind exists."""
    t = CFG["totals"]["env"]
    if roof in ("dome", "closed", "retractable"):
        return t["dome"]
    if wind_mph is None:
        return t["outdoor_base"]
    return t["outdoor_base"] - t["per_mph"] * (wind_mph - t["mean_wind"])


def halfpoint_ev(origin, market):
    """Expected edge (probability mass) of origin vs market spread using the
    empirical integer landing-mass table (key numbers)."""
    m = {int(k): v for k, v in CFG["market"]["mass"].items()}
    lo, hi = sorted((abs(origin), abs(market)))
    if origin * market < 0:  # opposite favorites -> straddles zero; treat magnitude sum
        lo, hi = 0.0, abs(origin) + abs(market)
    edge = sum(m.get(k, m[15]) for k in range(int(math.floor(lo)) + 1, int(math.ceil(hi))) if lo < k < hi)
    return round(edge, 3)


def main():
    spec = {g["game_id"]: g for g in json.loads((RUN / "final.json").read_text())}
    adj = {(a["away"], a["home"]): a for a in json.loads((RUN / "adjustments.json").read_text())["games"]}
    slate = load_slate()
    nf_spreads, nf_pts, nf_mods = load_nfelo()
    pffr, cole = load_pff_ratings(), load_cole_ratings()
    hfa = CFG["spreads"]["hfa_rating_paths"]
    W = CFG["spreads"]["weights"]
    out = []
    for s in slate:
        a, h = s["away"], s["home"]; key = (a, h)
        sp = spec[s["game_id"]]
        f = FEAT[key]
        # ---- spreads: same blend, research HFA on the rating paths
        sn = nf_spreads[key]["spread_nfelo"]
        hfa_g = CFG["spreads"].get("hfa_neutral", 0.5) if s["location"] != "Home" else hfa
        spff = -(pffr[h]["psr"] - pffr[a]["psr"]) - hfa_g
        scole = -(cole[h]["pr"] - cole[a]["pr"]) - hfa_g
        diff = spff - sn
        if abs(diff) > CFG["spreads"]["structural_clamp"]:
            spff = sn + math.copysign(CFG["spreads"]["structural_clamp"], diff)
        tier_a = (W["nfelo"] * sn + W["pff"] * spff) / (W["nfelo"] + W["pff"])
        core = W["nfelo"] * sn + W["pff"] * spff + W["cole"] * scole
        if abs(core - tier_a) > CFG["spreads"]["sleeve_clamp"]:
            core = tier_a + math.copysign(CFG["spreads"]["sleeve_clamp"], core - tier_a)
        # rest differential (research: applied once to the blend; Week 1 exempt)
        rest_adj = 0.0
        if CFG["schedule"]["rest_per_day"] and s["gameday"] and int(s["game_id"].split("_")[1]) > 1:
            rest_adj = -CFG["schedule"]["rest_per_day"] * max(-7, min(7, s["home_rest"] - s["away_rest"]))
        # ---- totals: research model V3
        T = CFG["totals"]
        total_core = (T["league_prior"] + T["b_elo"] * float(f["elo_sum"]) + T["b_pf"] * float(f["pf_dev"])
                      + T["b_pa"] * float(f["pa_dev"]) + T["b_qb"] * float(f["qb_sum"])
                      + T["b_div"] * int(f["div_game"]) + env_adj(s["roof"]))
        eff_adj = 0.0
        if T.get("b_eff"):
            eff_adj = max(-T["eff_cap"], min(T["eff_cap"], T["b_eff"] * float(f["epa_sum_dev"])))
        total_core += eff_adj
        # ---- audited §5 adjustments, with Week-1 caps from research
        A = adj[key]
        caps = CFG["week1_caps"] if int(s["game_id"].split("_")[1]) == 1 else CFG["caps"]
        scale = caps.get("scale", 1.0)
        adj_spread = max(-caps["spread"], min(caps["spread"], scale * sum(x["points"] for x in A["spread_adjustments"])))
        adj_total = max(-caps["total"], min(caps["total"], scale * sum(x["points"] for x in A["total_adjustments"])))
        spread_raw = core + rest_adj + adj_spread
        total_raw = total_core + adj_total
        spread_pub, total_pub = round_half(spread_raw), round_half(total_raw)
        # ---- team totals: identity, reallocation capped
        cap = CFG["team_totals"]["realloc_cap"]
        h1 = total_raw / 2 - spread_raw / 2; a1 = total_raw / 2 + spread_raw / 2
        hs = max(-cap, min(cap, sum(m["points"] for m in A["tt_modifiers"] if m["team"] == h)))
        as_ = max(-cap, min(cap, sum(m["points"] for m in A["tt_modifiers"] if m["team"] == a)))
        h1, a1 = h1 + hs, a1 + as_
        drift = (h1 + a1) - total_raw; h1 -= drift / 2; a1 -= drift / 2
        h_pub, a_pub = round_half(h1), round_half(a1)
        if abs(h_pub + a_pub - total_pub) > 1e-9:
            if abs(h_pub - h1) <= abs(a_pub - a1):
                a_pub = total_pub - h_pub
            else:
                h_pub = total_pub - a_pub
        # ---- confidence: market-gap driver (diagnostic use of the market only)
        U = CFG["uncertainty"]
        D_s = abs(spread_pub - sp["market_spread"]); D_t = abs(total_pub - sp["market_total"])
        sig_s = math.sqrt(U["base_spread"] ** 2 + (U["c"] * D_s) ** 2)
        sig_t = math.sqrt(U["base_total"] ** 2 + (U["c"] * D_t) ** 2)
        def tag(D, th):
            return "HIGH" if D <= th[0] else ("MED" if D <= th[1] else "LOW")
        out.append({
            "game_id": s["game_id"], "away": a, "home": h,
            "spread_nfelo": sn, "spread_pff_r": round(spff, 3), "spread_cole_r": round(scole, 3), "hfa_used": hfa_g,
            "spread_core_r": round(core, 3), "rest_adj": rest_adj, "adj_spread": round(adj_spread, 2),
            "spread_origin_r": spread_pub, "spread_origin_spec": sp["spread_origin"],
            "total_core_r": round(total_core, 3), "eff_adj": round(eff_adj, 2), "adj_total": round(adj_total, 2),
            "total_origin_r": total_pub, "total_origin_spec": sp["total_origin"],
            "tt_home_r": h_pub, "tt_away_r": a_pub, "tt_home_spec": sp["tt_home"], "tt_away_spec": sp["tt_away"],
            "market_spread": sp["market_spread"], "market_total": sp["market_total"],
            "gap_spread": round(spread_pub - sp["market_spread"], 1), "gap_total": round(total_pub - sp["market_total"], 1),
            "halfpoint_edge_vs_market": halfpoint_ev(spread_pub, sp["market_spread"]),
            "sigma_spread": round(sig_s, 2), "sigma_total": round(sig_t, 2),
            "conf_spread_r": tag(D_s, U["spread_thresholds"]), "conf_total_r": tag(D_t, U["total_thresholds"]),
            "conf_spread_spec": sp["conf_spread"], "conf_total_spec": sp["conf_total"],
            "features": {k: f[k] for k in ("elo_sum", "pf_dev", "pa_dev", "qb_sum", "div_game", "env", "epa_sum_dev")},
        })
        assert abs((h_pub + a_pub) - total_pub) < 1e-9
    (RUN / "final_research.json").write_text(json.dumps(out, indent=1))
    print(f"{'game':<9}{'spec S':>8}{'res S':>7} | {'spec T':>7}{'res T':>7} | {'TT res':>11} | {'mkt S/T':>12} | gapS gapT | edge | conf(res)")
    for g in out:
        print(f"{g['away']}@{g['home']:<5}{g['spread_origin_spec']:>+8.1f}{g['spread_origin_r']:>+7.1f} | {g['total_origin_spec']:>7.1f}{g['total_origin_r']:>7.1f} | "
              f"{g['tt_home_r']:>5.1f}/{g['tt_away_r']:<5.1f} | {g['market_spread']:>+5.1f}/{g['market_total']:<5.1f} | {g['gap_spread']:>+4.1f} {g['gap_total']:>+4.1f} | {g['halfpoint_edge_vs_market']:.3f} | {g['conf_spread_r'][0]}/{g['conf_total_r'][0]}")


if __name__ == "__main__":
    main()
