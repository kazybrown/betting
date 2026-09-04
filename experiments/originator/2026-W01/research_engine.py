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
ELO_BASE = {norm_team(r["team"]): (float(r["nfelo_base"]) - 1505.0) / 25.0
            for r in csv.DictReader(open(RAW / "elo_snapshot.csv"))}  # QB-free rating, points vs avg
INTERNATIONAL = ("Melbourne", "Wembley", "Tottenham", "Twickenham", "Frankfurt", "Munich", "Allianz", "Azteca",
                 "Estadio", "Corinthians", "Bernab", "Croke", "Dublin", "Madrid", "Berlin", "Olympiastadion")


def env_adj(roof, wind_mph=None, temp_f=None):
    """Environment table (totals): dome/closed/retractable-closed = +2.0; outdoor =
    -0.5 at league-average wind, -0.18 per mph above 8.4 when a verified forecast
    exists; <20F -1.0 (provisional); precipitation 0."""
    t = CFG["totals"]["env"]
    if roof in ("dome", "closed", "retractable"):
        return t["dome"]
    v = t["outdoor_base"] if wind_mph is None else t["outdoor_base"] - t["per_mph"] * (wind_mph - t["mean_wind"])
    if temp_f is not None and temp_f < 20:
        v += t["cold_lt20F"]
    return v


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
        if s["location"] == "Home":
            hfa_g = hfa
        elif any(k in (s["venue"] or "") for k in INTERNATIONAL):
            hfa_g = CFG["spreads"]["hfa_neutral_international"]
        else:
            hfa_g = CFG["spreads"]["hfa_neutral_domestic"]
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
        # ---- totals: synthesis formula, three implied paths blended .38/.32/.30
        T = CFG["totals"]
        week = int(s["game_id"].split("_")[1])
        LG = T["L_prev"] + T["week1_prior_offset"] if week == 1 else T["L_prev"] + T["median_target"]  # in-season blend needs L_ytd (n=0 in W1)
        DIV = T["div"]["div_game"] if s["div_game"] else T["div"]["non_div"]
        ENV = env_adj(s["roof"])
        R_h, R_a = ELO_BASE[h], ELO_BASE[a]                      # QB-free nfelo rating, points vs avg
        gt_h = float(f["pf_home"]) + float(f["pa_home"]); gt_a = float(f["pf_away"]) + float(f["pa_away"])  # prior-season team game-total means (K-blend is identity at gp=0)
        GT_dev = gt_h + gt_a - 2 * (LG + 0.5)
        np_ = T["nfelo_path"]
        t_nfelo = LG + np_["b_R"] * (R_h + R_a) + np_["b_gt"] * GT_dev + np_["b_qb"] * float(f["qb_sum"]) + DIV + ENV
        rp = T["rating_paths"]["b_R"]
        t_pff = LG + rp * (pffr[h]["psr"] + pffr[a]["psr"]) + DIV + ENV
        t_cole = LG + rp * (cole[h]["pr"] + cole[a]["pr"]) + DIV + ENV
        B = T["blend"]
        total_core = B["pff_implied"] * t_pff + B["nfelo_implied"] * t_nfelo + B["cole_implied"] * t_cole
        eff_adj = max(-T["eff"]["cap"], min(T["eff"]["cap"], T["eff"]["b_eff"] * float(f["epa_sum_dev"])))
        total_core += eff_adj
        totals_detail = {"LG": LG, "DIV": DIV, "ENV": ENV, "GT_dev": round(GT_dev, 2), "R_sum_qbfree": round(R_h + R_a, 2),
                         "t_nfelo": round(t_nfelo, 2), "t_pff": round(t_pff, 2), "t_cole": round(t_cole, 2)}
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
        h1 = total_raw / 2 - spread_raw / 2; a1 = total_raw / 2 + spread_raw / 2
        # synthesis change 6: the audited modifiers are all pace/explosive matchup reallocations -> zeroed
        realloc_dropped = sum(abs(m["points"]) for m in A["tt_modifiers"]) / 2
        h_pub, a_pub = round_half(h1), round_half(a1)
        if abs(h_pub + a_pub - total_pub) > 1e-9:
            if abs(h_pub - h1) <= abs(a_pub - a1):
                a_pub = total_pub - h_pub
            else:
                h_pub = total_pub - a_pub
        # ---- confidence: market-gap driver (diagnostic use of the market only)
        U = CFG["uncertainty"]
        D_s = abs(spread_pub - sp["market_spread"]); D_t = abs(total_pub - sp["market_total"])
        sig_s = math.sqrt(U["base_spread"] ** 2 + D_s ** 2)
        sig_t = math.sqrt(U["base_total"] ** 2 + D_t ** 2)
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
            "expected_real_gap_spread": round(U["gap_shrink_lambda"] * (spread_pub - sp["market_spread"]), 2),
            "excess_rmse_spread": round(sig_s - U["base_spread"], 2), "excess_rmse_total": round(sig_t - U["base_total"], 2),
            "totals_detail": totals_detail, "realloc_dropped_pts": realloc_dropped,
            "sigma_spread": round(sig_s, 2), "sigma_total": round(sig_t, 2),
            "conf_spread_r": tag(D_s, U["spread_thresholds"]), "conf_total_r": tag(D_t, U["total_thresholds"]),
            "conf_spread_spec": sp["conf_spread"], "conf_total_spec": sp["conf_total"],
            "features": {k: f[k] for k in ("elo_sum", "pf_dev", "pa_dev", "qb_sum", "div_game", "env", "epa_sum_dev")},
        })
        assert abs((h_pub + a_pub) - total_pub) < 1e-9
    (RUN / "final_research.json").write_text(json.dumps(out, indent=1))
    print(f"{'game':<9}{'spec S':>8}{'res S':>7} | {'spec T':>7}{'res T':>7} | {'TT res':>11} | {'mkt S/T':>12} | gapS gapT | realgap | xsRMSE S/T | conf(res)")
    for g in out:
        print(f"{g['away']}@{g['home']:<5}{g['spread_origin_spec']:>+8.1f}{g['spread_origin_r']:>+7.1f} | {g['total_origin_spec']:>7.1f}{g['total_origin_r']:>7.1f} | "
              f"{g['tt_home_r']:>5.1f}/{g['tt_away_r']:<5.1f} | {g['market_spread']:>+5.1f}/{g['market_total']:<5.1f} | {g['gap_spread']:>+4.1f} {g['gap_total']:>+4.1f} | {g['expected_real_gap_spread']:>+5.2f} | {g['excess_rmse_spread']:.2f}/{g['excess_rmse_total']:.2f} | {g['conf_spread_r'][0]}/{g['conf_total_r'][0]}")


if __name__ == "__main__":
    main()
