#!/usr/bin/env python3
"""Merge cores.json + sweep.json into per-game evidence bundles for the
adjust/verify workflow. Writes bundles.json = {rules, games:[...]}."""

import json
from pathlib import Path

from originator_engine import ELO_PER_POINT, norm_team

RUN = Path(__file__).resolve().parent

RULES = """SPEC CONTEXT (ORIGINATOR, NFL 2026 Week 1 — the card is originated, market lines are reference only, never an input):
- Spread convention: home perspective, NEGATIVE = home favored. Team totals: home_tt + away_tt = game_total.
- The blended cores already contain: nfelo's published spread (which itself bakes in HFA, rest, travel, surface, divisional history, AND per-QB starter adjustments — the per-game nfelo QB adjustment in points is provided) and, where present, a PFF/structural spread and a light TPT panel.
- §5 context adjustments are applied AFTER the blend, in this order, each needing a NAMED evidence item with a source. Allowed categories and ranges:
  1. qb_change: only for news the ratings did NOT bake (new injury/benching after 2026-08-31). Starter->backup: -2.0..-4.5 on that team. A WR1/skill absence is -0.5..-1.5 on that team's TEAM TOTAL, not automatically the spread.
  2. ol_pass_rush_mismatch: elite rush vs injured/weak OT: -1.0..-2.0 victim team total, -0.5..-1.5 spread toward the rush team.
  3. rest_short_week: Week 1 = everyone on 7 days; no adjustment unless evidence says otherwise.
  4. travel_international: west-to-east 10am-local kicks: -0.4..-0.8 visitor. International HFA already handled by nfelo — do not re-add.
  5. motivation: Weeks 15-18 only — NOT applicable in Week 1.
  6. weather: totals mostly; spread tweak only if wind/rain clearly hits one offense more (-0.5..-1.0). No verified forecast => no weather adjustment (climatology alone doesn't move numbers; flag instead).
- Hard caps on the SUM: spread ±2.5, total ±3.0 vs core. If you want more, the ratings are wrong — flag it instead of piling on.
- §6 team-total modifiers REALLOCATE points between the two team totals (they hardly change the game total): explosive offense vs weak explosive-play defense +0.5..+1.5 / opponent -0.5..-1.0; elite red-zone defense -0.5..-1.0 opponent; pace-possession mismatch ±0.5. Team totals are derived from the identity first; modifiers are on top.
- NEVER: anchor toward the market number, invent injuries, double-count what nfelo baked (its QB and HFA mods are in the bundle — e.g. if nfelo already knocked a team for a backup QB, a further qb_change adjustment needs NEW post-8/31 news), or move a number on vibes.
- BUILD (user instruction 2026-09-04): THREE ENGINES ONLY — nfelo (site model spread), PFF (point-spread rating) and Kevin Cole (Unexpected Points power rating, as of 9/1/26). Cole occupies the sleeve slot (spread .15 / total .30) with the single-source clamp; nfelo .46 / PFF .39 keep Tier A. The TPT panel (Donchess/FF-Winners) is DIAGNOSTIC ONLY (weight 0): mention where it sits vs the core, never treat it as an input. TOTALS: no engine publishes a total — every total is a §3.2-derived implied total from net ratings (nfelo-, PFF-, Cole-implied, blended .32/.38/.30); say so, and treat total confidence as LOW by policy.
- A previously AUDITED adjustment set for this game (prior_audited_adjustments_v2) is in the bundle. Keep it as your baseline: reproduce those adjustments unless the new PFF ratings change the evidence picture (e.g. an ol_pass_rush_mismatch magnitude, or a QB-rating-based backup valuation). State explicitly in origin_note whether the baseline was kept or changed and why. Rewrite the brief and origin_note so every Tier-A number cited is the CURRENT one (PFF point-spread ratings, not the old rank model).
- Voice: cold, precise, audit-ready; numbers first; no hype."""


def main():
    games = json.loads((RUN / "cores.json").read_text())
    sweep = json.loads((RUN / "sweep.json").read_text())

    inj_by_game = {}
    for blob in sweep.get("injuries", []):
        for g in (blob or {}).get("games", []):
            a, h = norm_team(g.get("away")), norm_team(g.get("home"))
            if a and h:
                inj_by_game[(a, h)] = g

    qb_by_team = {}
    for section in ("qbsAfc", "qbsNfc"):
        for t in (sweep.get(section) or {}).get("teams", []):
            tm = norm_team(t.get("team"))
            if tm:
                qb_by_team[tm] = t

    wx_by_game = {}
    for g in (sweep.get("weather") or {}).get("games", []):
        a, h = norm_team(g.get("away")), norm_team(g.get("home"))
        if a and h:
            wx_by_game[(a, h)] = g

    qual_by_team = {}
    for t in (sweep.get("quality") or {}).get("teams", []):
        tm = norm_team(t.get("team"))
        if tm:
            qual_by_team[tm] = t

    units_by_team = {}
    for t in (sweep.get("pffUnits") or {}).get("teams", []):
        tm = norm_team(t.get("team"))
        if tm:
            units_by_team[tm] = t

    power_by_team = {}
    for t in (sweep.get("pffPower") or {}).get("teams", []):
        tm = norm_team(t.get("team"))
        if tm:
            power_by_team[tm] = t

    prior_path = RUN / "adjustments_v4.json"
    prior = {}
    if prior_path.exists():
        for pg in json.loads(prior_path.read_text())["games"]:
            prior[(pg["away"], pg["home"])] = {
                "spread_adjustments": pg["spread_adjustments"],
                "total_adjustments": pg["total_adjustments"],
                "tt_modifiers": pg["tt_modifiers"],
                "considered_but_zero": pg["considered_but_zero"],
                "flags": pg["flags"],
            }

    bundles = []
    for g in games:
        a, h = g["away"], g["home"]
        qb_adj = g.get("nfelo_qb_adj") or {}
        bundle = {
            "away": a, "home": h,
            "kick": f"{g['weekday']} {g['gameday']} {g['gametime']} ET",
            "venue": g["venue"], "roof": g["roof"], "surface": g["surface"],
            "neutral_site": g["location"] != "Home", "div_game": g["div_game"],
            "rest": {"away": g["away_rest"], "home": g["home_rest"]},
            "cores": {
                "spread_nfelo": g["spread_nfelo"],
                "spread_pff": g["spread_pff"], "pff_meta": g["pff_meta"],
                "spread_tpt": g["spread_tpt"], "tpt_spread_detail": g["tpt_spread_detail"],
                "spread_core": g["spread_core"], "spread_weights": g["spread_weights"],
                "total_nfelo": g["total_nfelo"], "total_pff": g["total_pff"],
                "total_pff_basis": g["total_pff_basis"],
                "total_tpt": g["total_tpt"], "tpt_total_detail": g["tpt_total_detail"],
                "total_core": g["total_core"], "total_weights": g["total_weights"],
                "total_dvoa_diagnostic": g["total_dvoa_diag"],
                "home_wp_nfelo": g["home_wp_nfelo"],
            },
            "nfelo_baked_in": {
                "hfa_points": g["hfa_pts_nfelo"],
                "home_qb_adj_points": (round(qb_adj["home_elo"] / ELO_PER_POINT, 2)
                                        if qb_adj.get("home_elo") is not None else None),
                "away_qb_adj_points": (round(qb_adj["away_elo"] / ELO_PER_POINT, 2)
                                        if qb_adj.get("away_elo") is not None else None),
                "note": "positive = that team's current QB rates above its team baseline; from the 2026-09-01 repo snapshot",
                "spread_nfelo_repo_snapshot": g.get("nfelo_provenance", {}).get("spread_nfelo_repo_snapshot"),
                "spread_nfelo_site_pasted": g["spread_nfelo"],
                "site_minus_snapshot": g.get("nfelo_provenance", {}).get("nfelo_site_delta"),
                "WARNING": "spread_nfelo is the nfeloapp.com site value pasted 2026-09-02 (authoritative). The QB/HFA modifiers above are from the 09-01 repo snapshot; where site_minus_snapshot is non-zero the site has moved since the snapshot (possible QB/roster news nfelo has now baked) -- reason about what is baked from the site number, and never double count. nfelo publishes NO total; total_nfelo is a §3.2-derived nfelo-IMPLIED total.",
            },
            "injuries": inj_by_game.get((a, h), {"note": "no injury sweep data for this game"}),
            "qb_context": {"away": qb_by_team.get(a), "home": qb_by_team.get(h)},
            "weather": wx_by_game.get((a, h), {"note": "no weather sweep data"}),
            "quality_2025": {"away": qual_by_team.get(a), "home": qual_by_team.get(h)},
            "pff_units": {"away": units_by_team.get(a), "home": units_by_team.get(h)},
            "pff_power_rank_diag": {"away": power_by_team.get(a), "home": power_by_team.get(h),
                                     "note": "March post-FA web-recovered ranks; DIAGNOSTIC ONLY now that the authoritative PFF table is in pff_ratings"},
            "pff_ratings": {
                "source": "PFF Power Rankings table (pff.com/betting/nfl-power-rankings) pasted by the user 2026-09-01 — AUTHORITATIVE for this run (§12)",
                "convention": "Point Spread Rating = points vs league average, QB component included; spread_pff = -(psr_home - psr_away) - HFA",
                "home": g.get("pff_psr", {}).get("home"), "away": g.get("pff_psr", {}).get("away"),
                "home_qb_rating": g.get("pff_psr", {}).get("home_qb"), "away_qb_rating": g.get("pff_psr", {}).get("away_qb"),
                "home_proj_wins": g.get("pff_psr", {}).get("home_proj_wins"), "away_proj_wins": g.get("pff_psr", {}).get("away_proj_wins"),
            },
            "prior_audited_adjustments_v4": prior.get((a, h)),
            "dvoa_projection_2026": g.get("dvoa_proj"),
            "market_REFERENCE_ONLY_do_not_anchor": {
                "spread_current": g["market_spread"], "total_current": g["market_total"],
                "spread_open": g.get("market_open_spread"), "total_open": g.get("market_open_total"),
            },
            "kevin_cole": {
                "source": "Unexpected Points Subscriber Data (Kevin Cole) '2026 Power Rankings' tab, as of 9/1/26, read via Google Drive 2026-09-04 — third rating engine (sleeve slot: spread .15 / total .30)",
                "pr_home": g["cole_meta"].get("pr_home"), "pr_away": g["cole_meta"].get("pr_away"),
                "betting_pr_home_diag": g["cole_meta"].get("betting_pr_home"), "betting_pr_away_diag": g["cole_meta"].get("betting_pr_away"),
                "spread_cole": g["spread_cole"], "spread_from_betting_pr_diag": g["cole_meta"].get("spread_betting_pr_diag"),
                "total_cole_implied": g["total_cole_implied"],
            },
            "totals_basis": {"note": g.get("total_basis"), "nfelo_implied": g["total_nfelo"], "pff_implied": g["total_pff_implied"],
                             "cole_implied": g["total_cole_implied"], "confidence_policy": "forced LOW — no engine publishes a total"},
            "tpt_panel": {
                "ROLE": "DIAGNOSTIC ONLY in this build (weight 0) per user instruction 2026-09-04; report its gap vs the origin core, never blend it",
                "gaps_vs_core": g["tpt_diag"],
                "spread_systems": g["tpt_spread_detail"], "spread_panel_value": g["spread_tpt"],
                "total_systems": g["tpt_total_detail"], "total_panel_value": g["total_tpt"],
                "systems_blank_in_tpt_file": {"spreads": ["PIR", "STJ"], "totals": ["RPXL", "RWP", "DOK"]},
                "system_clamps": g.get("tpt_system_clamps", []),
                "donc_source": next((v.get("source") for v in sweep.get("tptSpreads", {}).get("values", [])
                                     if v.get("system") == "DONC" and norm_team(v.get("away")) == a and norm_team(v.get("home")) == h), None),
                "dratings_team_totals_diag": next(({"tt_away": d["tt_away"], "tt_home": d["tt_home"]} for d in sweep.get("dratingsPasted", {}).get("games", [])
                                                   if d["away"] == a and d["home"] == h), None),
            },
            "uncertainty": {"spread_sd": g["spread_sd"], "total_sd": g["total_sd"],
                             "n_spread_sources": g["n_spread_sources"],
                             "n_total_sources": g["n_total_sources"]},
        }
        bundles.append(bundle)

    out = {"rules": RULES, "games": bundles}
    (RUN / "bundles.json").write_text(json.dumps(out, indent=1))
    print(f"bundles.json written: {len(bundles)} games")


if __name__ == "__main__":
    main()
