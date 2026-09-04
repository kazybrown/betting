#!/usr/bin/env python3
"""Research-calibrated ORIGINATOR engine (v6): applies the parameter changes the
expert panel upheld (research_config.json, research/synthesis.json) on top of
the spec engine's loaders and audited adjustments.

Stages (mirror originator_engine.py so the downstream tooling is unchanged):
  --stage cores : run the spec cores (kept as cores_spec.json), then rebuild
                  spreads/totals with the research parameters -> cores.json
  --stage final : apply adjustments.json with the research caps, identity
                  team totals (matchup reallocations zeroed by the panel),
                  market-distance confidence tags -> final.json
Publishing stays in originator_engine.py --stage publish.
"""

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
from originator_engine import (RAW, build_cores as spec_build_cores, round_half, norm_team)  # noqa: E402

CFG = json.loads((RUN / "research_config.json").read_text())
FEAT = {(r["away"], r["home"]): r for r in csv.DictReader(open(RUN / "research" / "week1_features.csv"))}
ELO_BASE = {norm_team(r["team"]): (float(r["nfelo_base"]) - 1505.0) / 25.0
            for r in csv.DictReader(open(RAW / "elo_snapshot.csv"))}  # QB-free nfelo rating, points vs avg
INTERNATIONAL = ("Melbourne", "Wembley", "Tottenham", "Twickenham", "Frankfurt", "Munich", "Allianz", "Azteca",
                 "Estadio", "Corinthians", "Bernab", "Croke", "Dublin", "Madrid", "Berlin", "Olympiastadion")
WEEK = 1
CLIP = lambda x, lo, hi: max(lo, min(hi, x))  # noqa: E731

METHOD = {
    "version": "v6 (research-calibrated, 2026-09-04)",
    "hfa_rating_paths": f"PFF and Cole spreads use HFA {CFG['spreads']['hfa_rating_paths']} (neutral: international "
                        f"{CFG['spreads']['hfa_neutral_international']}, domestic {CFG['spreads']['hfa_neutral_domestic']}); "
                        "nfelo's per-game hfa_mod is never applied to the rating paths (panel: hfa)",
    "spread_blend": "spread_core = .46 nfelo + .39 PFF + .15 Cole; structural clamp 4.5 on PFF vs nfelo; §1 single-computer clamp on the sleeve: "
                    "the Cole sleeve may move the blend at most 1.0 from the Tier-A (nfelo/PFF) blend (|core − Tier-A| ≤ 1.0, as implemented in v5); "
                    "Cole's raw distance from the Tier-A blend is not itself clamped",
    "rest": f"rest differential priced once on the blend at -{CFG['schedule']['rest_per_day']}/day (clip ±{CFG['schedule']['rest_clip_days']}), Week 1 exempt; "
            "short-week / bye / west-to-east clauses deleted (panel: schedule)",
    "totals_prior": f"Week-1 league prior LG = {CFG['totals']['L_prev']} {CFG['totals']['week1_prior_offset']:+.1f} = "
                    f"{CFG['totals']['L_prev'] + CFG['totals']['week1_prior_offset']:.1f} (panel: totals/earlyseason; in-season: blended prior -0.5)",
    "total_nfelo_implied": "LG + 0.10·(R_home + R_away, QB-free nfelo Elo in points) + 0.30·GT_dev (prior-season PF+PA vs league) "
                           "+ 0.72·QB_sum (nfelo QB adj, points) + DIV + ENV",
    "total_pff_implied": "LG + 0.30·(PSR_home + PSR_away) + DIV + ENV",
    "total_cole_implied": "LG + 0.30·(PR_home + PR_away) + DIV + ENV",
    "div_env": f"DIV {CFG['totals']['div']['div_game']} divisional / {CFG['totals']['div']['non_div']:+.2f} non-divisional; ENV dome/closed +{CFG['totals']['env']['dome']}, "
               f"outdoor {CFG['totals']['env']['outdoor_base']} at league-average wind (-{CFG['totals']['env']['per_mph']}/mph above {CFG['totals']['env']['mean_wind']} only with a verified forecast), "
               f"<20°F {CFG['totals']['env']['cold_lt20F']}, precipitation 0",
    "total_blend": "total_core = .38 PFF-implied + .32 nfelo-implied + .30 Cole-implied + EFF, EFF = clip(3.0·epa_sum_dev, ±2.0) "
                   "(prior-season EPA/play efficiency, both teams, vs league); pace and style adjustments 0 (panel: pace)",
    "caps": f"§5 caps Week 1: spread ±{CFG['week1_caps']['spread']}, total ±{CFG['week1_caps']['total']} (in-season ±{CFG['caps']['spread']}/±{CFG['caps']['total']})",
    "team_totals": "identity only (home = T/2 - S/2, away = T/2 + S/2); §6 matchup reallocations (explosive, red-zone, pace, PF/PA profile) "
                   "zeroed by the panel (team totals: overturned); any future reallocation capped ±0.5 with spread coherence",
    "qb": "stint penalty 2.5 (1st start) / 1.5 (2nd-3rd) / 0.5 (4th+) applied NET of what the inputs embed; planned offseason starters 0",
    "confidence": f"tags = distance of the published number to the latest market line (diagnostic; the market is never an input): "
                  f"spread HIGH <{CFG['uncertainty']['spread_thresholds'][0]}, MED to {CFG['uncertainty']['spread_thresholds'][1]}, LOW beyond; "
                  f"total HIGH <{CFG['uncertainty']['total_thresholds'][0]}, MED to {CFG['uncertainty']['total_thresholds'][1]}, LOW beyond; "
                  f"sigma = sqrt(BASE² + D²), BASE {CFG['uncertainty']['base_spread']}/{CFG['uncertainty']['base_total']}; "
                  f"expected real share of a gap λ = {CFG['uncertainty']['gap_shrink_lambda']}",
    "rounding": "half-up to .0/.5 (§8); no key-number rounding, no steam lean (panel: market)",
}


def env_adj(roof, wind_mph=None, temp_f=None):
    t = CFG["totals"]["env"]
    if roof in ("dome", "closed", "retractable"):
        return t["dome"]
    v = t["outdoor_base"] if wind_mph is None else t["outdoor_base"] - t["per_mph"] * (wind_mph - t["mean_wind"])
    if temp_f is not None and temp_f < 20:
        v += t["cold_lt20F"]
    return v


def site_hfa(g):
    S = CFG["spreads"]
    if g["location"] == "Home":
        return S["hfa_rating_paths"], "home"
    if any(k in (g.get("venue") or "") for k in INTERNATIONAL):
        return S["hfa_neutral_international"], "neutral-international"
    return S["hfa_neutral_domestic"], "neutral-domestic"


def sd_range(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None, None
    return round(statistics.pstdev(vals), 3), round(max(vals) - min(vals), 2)


# ---------------------------------------------------------------- cores

def build_cores():
    spec = spec_build_cores()                       # writes cores.json (spec)
    shutil.copy(RUN / "cores.json", RUN / "cores_spec.json")
    S, W, T = CFG["spreads"], CFG["spreads"]["weights"], CFG["totals"]
    B = T["blend"]
    out = []
    for g0 in spec:
        g = dict(g0)
        a, h = g["away"], g["home"]
        f = FEAT[(a, h)]
        week = int(g["game_id"].split("_")[1])
        # ---- spreads
        sn = g["spread_nfelo"]
        hfa_g, hfa_kind = site_hfa(g)
        psr_h, psr_a = g["pff_psr"]["home"], g["pff_psr"]["away"]
        pr_h, pr_a = g["cole_meta"]["pr_home"], g["cole_meta"]["pr_away"]
        spff_raw = -(psr_h - psr_a) - hfa_g
        scole = -(pr_h - pr_a) - hfa_g
        spff, structural_flag = spff_raw, None
        if abs(spff_raw - sn) > S["structural_clamp"]:
            spff = sn + math.copysign(S["structural_clamp"], spff_raw - sn)
            structural_flag = (f"PFF spread {spff_raw:+.2f} differs from nfelo {sn:+.2f} by more than "
                               f"{S['structural_clamp']}; clamped to {spff:+.2f} (§3.1)")
        tier_a = (W["nfelo"] * sn + W["pff"] * spff) / (W["nfelo"] + W["pff"])
        core_raw = W["nfelo"] * sn + W["pff"] * spff + W["cole"] * scole
        core, sleeve_clamp = core_raw, None
        if abs(core_raw - tier_a) > S["sleeve_clamp"]:
            core = tier_a + math.copysign(S["sleeve_clamp"], core_raw - tier_a)
            sleeve_clamp = f"Cole sleeve moved the blend {core_raw - tier_a:+.2f} vs Tier A; clamped to ±{S['sleeve_clamp']} (§1)"
        rest_adj, rest_note = 0.0, "Week 1: exempt (panel: schedule)"
        if week > 1 and CFG["schedule"]["rest_per_day"]:
            d = CLIP(g["home_rest"] - g["away_rest"], -CFG["schedule"]["rest_clip_days"], CFG["schedule"]["rest_clip_days"])
            rest_adj = -CFG["schedule"]["rest_per_day"] * d
            rest_note = f"rest differential {d:+d} days × -{CFG['schedule']['rest_per_day']}"
        # ---- totals
        LG = T["L_prev"] + T["week1_prior_offset"] if week == 1 else T["L_prev"] + T["median_target"]
        DIV = T["div"]["div_game"] if g["div_game"] else T["div"]["non_div"]
        ENV = env_adj(g["roof"])
        R_h, R_a = ELO_BASE[h], ELO_BASE[a]
        gt_h = float(f["pf_home"]) + float(f["pa_home"]); gt_a = float(f["pf_away"]) + float(f["pa_away"])
        GT_dev = gt_h + gt_a - 2 * (LG + 0.5)
        qb_sum = float(f["qb_sum"])
        np_ = T["nfelo_path"]; rp = T["rating_paths"]["b_R"]
        t_nfelo = LG + np_["b_R"] * (R_h + R_a) + np_["b_gt"] * GT_dev + np_["b_qb"] * qb_sum + DIV + ENV
        t_pff = LG + rp * (psr_h + psr_a) + DIV + ENV
        t_cole = LG + rp * (pr_h + pr_a) + DIV + ENV
        eff = CLIP(T["eff"]["b_eff"] * float(f["epa_sum_dev"]), -T["eff"]["cap"], T["eff"]["cap"])
        total_core = B["pff_implied"] * t_pff + B["nfelo_implied"] * t_nfelo + B["cole_implied"] * t_cole + eff
        ssd, srng = sd_range([sn, spff, scole]); tsd, trng = sd_range([t_nfelo, t_pff, t_cole])
        g.update({
            "build": "v6 research-calibrated three-engine: nfelo + PFF + Kevin Cole (TPT diagnostic only)",
            "method": METHOD,
            "hfa_rating_paths": hfa_g, "hfa_kind": hfa_kind,
            "spread_pff": round(spff, 3), "spread_pff_unclamped": round(spff_raw, 3),
            "spread_cole": round(scole, 3),
            "spread_core": round(core + rest_adj, 3), "spread_core_before_rest": round(core, 3),
            "rest_adj": round(rest_adj, 3), "rest_note": rest_note,
            "structural_flag": structural_flag, "sleeve_clamp_note": sleeve_clamp,
            "spread_weights": {"nfelo": W["nfelo"], "pff": W["pff"], "cole": W["cole"], "tpt": 0.0},
            "total_nfelo": round(t_nfelo, 3), "total_pff_implied": round(t_pff, 3), "total_cole_implied": round(t_cole, 3),
            "total_pff": None, "total_pff_basis": None,
            "total_core": round(total_core, 3), "eff_adj": round(eff, 3),
            "total_weights": {"pff_implied": B["pff_implied"], "nfelo_implied": B["nfelo_implied"], "cole_implied": B["cole_implied"], "tpt": 0.0},
            "total_basis": ("all three totals are §3.2-derived implied totals on the research (v6) formula: "
                            f"LG {LG:.1f} + rating term + DIV {DIV:+.2f} + ENV {ENV:+.1f}; nfelo path adds GT_dev and QB terms; "
                            f"blend .38/.32/.30 + EFF {eff:+.2f}; no engine publishes a total"),
            "totals_detail": {"LG": LG, "DIV": DIV, "ENV": ENV, "roof": g["roof"], "R_home_qbfree": round(R_h, 3), "R_away_qbfree": round(R_a, 3),
                              "GT_home_prior": round(gt_h, 2), "GT_away_prior": round(gt_a, 2), "GT_dev": round(GT_dev, 3),
                              "qb_sum_pts": qb_sum, "epa_sum_dev": float(f["epa_sum_dev"]), "eff_adj": round(eff, 3),
                              "t_nfelo": round(t_nfelo, 3), "t_pff": round(t_pff, 3), "t_cole": round(t_cole, 3)},
            "spread_sd": ssd, "spread_range": srng, "total_sd": tsd, "total_range": trng,
            "n_spread_sources": 3, "n_total_sources": 3,
            "spec_reference": {"spread_pff_spec": g0["spread_pff"], "spread_cole_spec": g0["spread_cole"], "spread_core_spec": g0["spread_core"],
                               "total_core_spec": g0["total_core"], "hfa_pts_nfelo_spec": g0.get("hfa_pts_nfelo")},
        })
        g["pff_meta"] = dict(g["pff_meta"], hfa_used=hfa_g, hfa_basis="research HFA on rating paths (v6)", spread_unclamped=round(spff_raw, 3))
        g["cole_meta"] = dict(g["cole_meta"], hfa_used=hfa_g, hfa_basis="research HFA on rating paths (v6)")
        if g.get("tpt_diag"):
            g["tpt_diag"] = dict(g["tpt_diag"],
                                 spread_gap_vs_core=(round(g["tpt_diag"]["spread_panel"] - g["spread_core"], 2) if g["tpt_diag"].get("spread_panel") is not None else None),
                                 total_gap_vs_core=(round(g["tpt_diag"]["total_panel"] - g["total_core"], 2) if g["tpt_diag"].get("total_panel") is not None else None))
        out.append(g)
    (RUN / "cores.json").write_text(json.dumps(out, indent=1))
    print("\ncores.json (v6 research) written")
    print(f"{'game':<9}{'nfelo':>7}{'PFF':>8}{'Cole':>8}{'core':>8} | {'T nf':>7}{'T pff':>7}{'T cole':>7}{'EFF':>6}{'core':>7} | spec S/T")
    for g in out:
        print(f"{g['away']}@{g['home']:<5}{g['spread_nfelo']:>+7.1f}{g['spread_pff']:>+8.2f}{g['spread_cole']:>+8.2f}{g['spread_core']:>+8.2f} | "
              f"{g['total_nfelo']:>7.2f}{g['total_pff_implied']:>7.2f}{g['total_cole_implied']:>7.2f}{g['eff_adj']:>+6.2f}{g['total_core']:>7.2f} | "
              f"{g['spec_reference']['spread_core_spec']:+.2f}/{g['spec_reference']['total_core_spec']:.2f}")
    return out


# ---------------------------------------------------------------- final

def market_ref(g):
    """Latest market line available for the diagnostic confidence tags."""
    ms = g.get("market_nfelo_current") if g.get("market_nfelo_current") is not None else g["market_spread"]
    ms_src = "nfelo site 'Current' (pasted 2026-09-04)" if g.get("market_nfelo_current") is not None else "nflverse games.csv"
    mt = g.get("market_tpt_total") if g.get("market_tpt_total") is not None else g["market_total"]
    mt_src = "TPT file line (pasted 2026-09-02)" if g.get("market_tpt_total") is not None else "nflverse games.csv"
    return ms, ms_src, mt, mt_src


def finalize():
    games = json.loads((RUN / "cores.json").read_text())
    adj_blob = json.loads((RUN / "adjustments.json").read_text())
    adj_by_key = {(a["away"], a["home"]): a for a in adj_blob["games"]}
    U = CFG["uncertainty"]
    out_rows = []
    for g in games:
        key = (g["away"], g["home"])
        A = adj_by_key.get(key, {})
        week = int(g["game_id"].split("_")[1])
        caps = CFG["week1_caps"] if week == 1 else CFG["caps"]
        s_adjs = A.get("spread_adjustments", [])
        t_adjs = A.get("total_adjustments", [])
        s_sum = sum(x["points"] for x in s_adjs); t_sum = sum(x["points"] for x in t_adjs)
        adj_spread = CLIP(caps["scale"] * s_sum, -caps["spread"], caps["spread"])
        adj_total = CLIP(caps["scale"] * t_sum, -caps["total"], caps["total"])
        spread_raw = g["spread_core"] + adj_spread
        total_raw = g["total_core"] + adj_total
        spread_pub, total_pub = round_half(spread_raw), round_half(total_raw)
        # §6 identity; panel change 6: matchup reallocations zeroed
        S_, T_ = spread_raw, total_raw
        h1, a1 = T_ / 2.0 - S_ / 2.0, T_ / 2.0 + S_ / 2.0
        tt_mods = A.get("tt_modifiers", [])
        zeroed = [dict(m, applied_points=0.0, zeroed_by="research panel (team totals: matchup reallocation overturned)") for m in tt_mods]
        h_pub, a_pub = round_half(h1), round_half(a1)
        total_nudged = False
        if abs((h_pub + a_pub) - total_pub) > 1e-9:
            if abs(h_pub - h1) <= abs(a_pub - a1):
                a_pub = total_pub - h_pub
            else:
                h_pub = total_pub - a_pub
            if min(a_pub, h_pub) < 0 or abs(a_pub * 2 - round(a_pub * 2)) > 1e-9:
                total_pub = total_pub + (0.5 if (h1 + a1) > total_pub else -0.5)
                a_pub = total_pub - h_pub
                total_nudged = True
        # §7 confidence: distance to the latest market line (diagnostic only)
        ms, ms_src, mt, mt_src = market_ref(g)
        D_s, D_t = abs(spread_pub - ms), abs(total_pub - mt)
        sig_s = math.sqrt(U["base_spread"] ** 2 + D_s ** 2); sig_t = math.sqrt(U["base_total"] ** 2 + D_t ** 2)
        tag = lambda D, th: "HIGH" if D < th[0] else ("MED" if D < th[1] else "LOW")  # noqa: E731
        conf_s, conf_t = tag(D_s, U["spread_thresholds"]), tag(D_t, U["total_thresholds"])
        g_out = dict(g)
        g_out.update({
            "adj_spread": round(adj_spread, 2), "adj_total": round(adj_total, 2),
            "adj_spread_uncapped": round(s_sum, 2), "adj_total_uncapped": round(t_sum, 2), "caps_applied": caps,
            "spread_adjustment_log": s_adjs, "total_adjustment_log": t_adjs,
            "tt_modifier_log": zeroed, "tt_modifiers_zeroed_by_panel": bool(tt_mods),
            "spread_raw": round(spread_raw, 3), "total_raw": round(total_raw, 3),
            "spread_origin": spread_pub, "total_origin": total_pub, "total_nudged_for_identity": total_nudged,
            "tt_home_raw": round(h1, 3), "tt_away_raw": round(a1, 3),
            "tt_home": h_pub, "tt_away": a_pub,
            "conf_spread": conf_s, "conf_total": conf_t,
            "conf_basis": {"driver": "distance to latest market line (diagnostic)", "market_spread_ref": ms, "market_spread_ref_source": ms_src,
                           "market_total_ref": mt, "market_total_ref_source": mt_src, "D_spread": round(D_s, 2), "D_total": round(D_t, 2),
                           "sigma_spread": round(sig_s, 2), "sigma_total": round(sig_t, 2),
                           "excess_rmse_spread": round(sig_s - U["base_spread"], 2), "excess_rmse_total": round(sig_t - U["base_total"], 2),
                           "expected_real_gap_spread": round(U["gap_shrink_lambda"] * (spread_pub - ms), 2),
                           "expected_real_gap_total": round(U["gap_shrink_lambda"] * (total_pub - mt), 2)},
            "conf_total_policy": None,
            "origin_note": A.get("origin_note", ""),
            "tt_split_note": A.get("tt_split_note", ""),
            "flags": A.get("flags", []),
        })
        assert abs((g_out["tt_home"] + g_out["tt_away"]) - g_out["total_origin"]) < 1e-9, g_out["game_id"]
        assert g_out["spread_origin"] % 0.5 == 0 and g_out["total_origin"] % 0.5 == 0
        out_rows.append(g_out)
    (RUN / "final.json").write_text(json.dumps(out_rows, indent=1))
    print("final.json (v6 research) written")
    for g in out_rows:
        cb = g["conf_basis"]
        print(f"  {g['away']}@{g['home']}: {g['home']} {g['spread_origin']:+.1f} | T {g['total_origin']} | TT {g['tt_home']}/{g['tt_away']} | "
              f"adj {g['adj_spread']:+.1f}/{g['adj_total']:+.1f} | {g['conf_spread']}/{g['conf_total']} | D {cb['D_spread']}/{cb['D_total']}"
              + (" | tt realloc zeroed" if g["tt_modifiers_zeroed_by_panel"] else ""))
    return out_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cores", "final"], required=True)
    args = ap.parse_args()
    build_cores() if args.stage == "cores" else finalize()
