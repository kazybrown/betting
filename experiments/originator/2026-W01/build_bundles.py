#!/usr/bin/env python3
"""Merge cores.json + sweep.json into per-game evidence bundles for the
adjust/verify workflow. Writes bundles.json = {rules, games:[...]}."""

import json
from pathlib import Path

from originator_engine import ELO_PER_POINT, norm_team

RUN = Path(__file__).resolve().parent

RULES = """SPEC CONTEXT (ORIGINATOR, NFL 2026 Week 1 — v6 RESEARCH-CALIBRATED build; the card is originated, market lines are reference only, never an input):
- Spread convention: home perspective, NEGATIVE = home favored. Team totals: home_tt + away_tt = game_total, from the identity only in v6 (see §6 below).
- ENGINES (user instruction 2026-09-04): THREE ONLY — nfelo (site model spread, bundle cores.spread_nfelo; it bakes in HFA, rest, travel, surface, divisional history AND per-QB starter adjustments — the per-game QB adjustment in points is in nfelo_baked_in), PFF (point-spread rating, bundle pff_ratings; spread_pff = -(PSR_home - PSR_away) - 1.75) and Kevin Cole (Unexpected Points power rating as of 9/1/26, bundle kevin_cole; spread_cole = -(PR_home - PR_away) - 1.75). Weights: nfelo .46 / PFF .39 / Cole .15; structural clamp 4.5 on PFF vs nfelo; Cole is the sleeve occupant with the §1 single-source clamp (1.0). The TPT panel (Donchess/FF-Winners, bundle tpt_panel) is DIAGNOSTIC ONLY — weight 0; cite its gap vs the core in one clause, never as an input.
- RESEARCH CHANGES ALREADY INSIDE THE CORES (10-expert panel synthesis 2026-09-04, bundle cores.method) — NEVER re-add any of them as a §5 adjustment: (1) HFA on the PFF/Cole paths is 1.75 (0.75 international neutral, 0.0 domestic neutral), no longer nfelo's per-game HFA; (2) totals are on the research formula: Week-1 league prior 45.0 + rating term + DIV (-0.85 divisional / +0.45 non-divisional) + ENV (dome/closed +2.0; outdoor -0.5 when no forecast exists) [+ prior-season game-total and QB terms on the nfelo path], blended .38 PFF-implied / .32 nfelo-implied / .30 Cole-implied, plus an efficiency term EFF = clip(3.0 × prior-season EPA/play deviation of both teams, ±2.0) — every component is in cores.totals_detail; (3) rest differential is priced once by the engine (Week 1 exempt: everyone on 7 days); (4) pace and style adjustments are 0 (no out-of-sample value).
- §5 context adjustments are applied AFTER the blend, each needing a NAMED evidence item with a source from the bundle. Categories and ranges in v6:
  1. qb_change: ONLY for a starter change the ratings did NOT bake (new injury/benching AFTER 2026-08-31, the nfelo data date; check nfelo_baked_in.site_minus_snapshot — a site number that moved may already carry it). Research stint table, applied NET of what the inputs already embed: replacement making his 1st career start -2.5, 2nd-3rd start -1.5, 4th+ start -0.5 on that team (this replaces the old -2.0..-4.5 range); a planned offseason starter (rookie or new starter named in camp) is 0 — the ratings already carry him. A WR1/skill absence is -0.5..-1.5 on that team's TEAM TOTAL, not automatically the spread.
  2. ol_pass_rush_mismatch: elite rush vs injured/weak OT: -1.0..-2.0 on the victim's team total, -0.5..-1.5 spread toward the rush team.
  3. rest_short_week: DELETED by the panel (rest is priced once in the engine; short-week, bye and west-to-east 10am body-clock clauses showed no out-of-sample value). Never file one.
  4. travel_international: DELETED for the same reason; international/neutral HFA is handled in the cores. Never file one.
  5. motivation: Weeks 15-18 only — NOT applicable in Week 1.
  6. weather: totals only, and ONLY with a verified forecast (wind: -0.18 per mph above 8.4; <20°F: -1.0; precipitation 0). NO verified forecast exists for any Week 1 game => no numeric weather adjustment is allowed; the outdoor -0.5 is already in ENV. Flag outdoor games instead.
- Hard caps on the SUM per game in Week 1 (research): spread ±2.0, total ±2.5 vs core (in-season ±2.5/±3.0). Needing more means the ratings are wrong — flag, don't pile on.
- §6 team totals: IDENTITY ONLY in v6 (home = T/2 - S/2, away = T/2 + S/2). The panel OVERTURNED matchup reallocations (explosive-play, red-zone, pace/possession, PF/PA profile): they had no out-of-sample value. tt_modifiers MUST be an empty list; the engine zeroes anything proposed. tt_split_note explains the identity skew only. A §5 item filed against a team total is applied to the game total and allocated by the identity.
- Confidence tags are set by the engine from the distance between the published number and the latest market line (diagnostic use only): spread HIGH <1.5 / MED <3.0 / LOW; total HIGH <2.5 / MED <5.0 / LOW. Do not write your own tag, and never anchor toward the market number.
- NEVER: anchor toward the market number, invent injuries, double-count what nfelo or the cores already bake (check nfelo_baked_in and cores.method), or move a number on vibes. Roster-status codes like A02 with "decode unverified" are WEAK evidence — at most a flag or a small team-total tweak with the uncertainty named, never a max-range adjustment.
- nfelo UPDATE: spread_nfelo is the nfeloapp.com site value pasted by the user 2026-09-02 (authoritative, §12). nfelo_baked_in shows the earlier repo-snapshot spread and site_minus_snapshot; a non-zero delta means nfelo moved after its 09-01 snapshot (plausibly QB/roster news it has now baked) — treat what the SITE number bakes as baked, never double count, and say so in the brief. nfelo publishes NO game total or projected score: every total on this card is a derived implied total — call them "implied (derived)" in the brief, never an nfelo/PFF/Cole total. Home WP for moved games is interpolated on nfelo's own line->WP curve (bundle cores.home_wp_nfelo).
- The bundle carries prior_audited_adjustments_v5 — this game's adjustment set from the previous (v5) pass, audited under the OLD rules. Treat it as the BASELINE but RE-BASE it on the v6 rules: keep evidence-backed qb_change / ol_pass_rush_mismatch / team-total items (re-sized to the stint table where a qb_change applies), DROP every rest/travel/pace/weather item and ALL tt_modifiers, and state in origin_note what was kept, re-sized or dropped and why. Every number in the brief must be the CURRENT v6 one from cores / pff_ratings / kevin_cole / totals_basis (the v5 numbers used a different HFA and totals formula — do not quote them).
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

    prior_path = RUN / "adjustments_v5.json"
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
                "spread_cole": g["spread_cole"], "hfa_rating_paths": g.get("hfa_rating_paths"), "hfa_kind": g.get("hfa_kind"),
                "structural_flag": g.get("structural_flag"), "sleeve_clamp_note": g.get("sleeve_clamp_note"),
                "rest_adj": g.get("rest_adj"), "rest_note": g.get("rest_note"),
                "method": g.get("method"), "totals_detail": g.get("totals_detail"), "eff_adj": g.get("eff_adj"),
                "spec_reference_v5": g.get("spec_reference"),
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
                "hfa_points": g["hfa_pts_nfelo"], "hfa_points_note": "nfelo's own HFA, inside spread_nfelo only; v6 uses 1.75 on the PFF/Cole paths (cores.hfa_rating_paths)",
                "home_qb_adj_points": (round(qb_adj["home_elo"] / ELO_PER_POINT, 2)
                                        if qb_adj.get("home_elo") is not None else None),
                "away_qb_adj_points": (round(qb_adj["away_elo"] / ELO_PER_POINT, 2)
                                        if qb_adj.get("away_elo") is not None else None),
                "note": "positive = that team's current QB rates above its team baseline; from the 2026-09-01 repo snapshot",
                "spread_nfelo_repo_snapshot": g.get("nfelo_provenance", {}).get("spread_nfelo_repo_snapshot"),
                "spread_nfelo_site_pasted": g["spread_nfelo"],
                "site_minus_snapshot": g.get("nfelo_provenance", {}).get("nfelo_site_delta"),
                "WARNING": "spread_nfelo is the nfeloapp.com site value pasted 2026-09-02 (authoritative). The QB/HFA modifiers above are from the 09-01 repo snapshot; where site_minus_snapshot is non-zero the site has moved since the snapshot (possible QB/roster news nfelo has now baked) -- reason about what is baked from the site number, and never double count. nfelo publishes NO total; total_nfelo is a derived nfelo-IMPLIED total on the v6 research formula (cores.totals_detail).",
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
            "prior_audited_adjustments_v5": prior.get((a, h)),
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
                             "cole_implied": g["total_cole_implied"], "components": g.get("totals_detail"),
                             "confidence_policy": "v6: engine sets the tag from the distance to the latest market line (diagnostic); no engine publishes a total"},
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
