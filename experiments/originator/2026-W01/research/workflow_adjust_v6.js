export const meta = {
  name: 'originator-adjust-w1-v6',
  description: 'v6 research-calibrated: per-game §5 adjustment analysts, adversarial auditors, and a completeness critic for the Week 1 origin card',
  phases: [
    { title: 'Adjust', detail: 'one analyst per game proposes evidence-named adjustments + brief' },
    { title: 'Verify', detail: 'one adversarial auditor per game tries to refute each adjustment' },
    { title: 'Critic', detail: 'completeness critic enforces the no-silent-injuries rule' },
  ],
}

// args = { games: [{away, home}, ...] }

const RULES = `SPEC CONTEXT (ORIGINATOR, NFL 2026 Week 1 — v6 RESEARCH-CALIBRATED build; the card is originated, market lines are reference only, never an input):
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
- Voice: cold, precise, audit-ready; numbers first; no hype.`

const READ_CMD = (away, home) =>
  `python3 -c "import json;d=json.load(open('/home/user/originator-2026-w01/bundles.json'));g=[x for x in d['games'] if x['away']=='${away}' and x['home']=='${home}'][0];print(json.dumps(g,indent=1))"`

const ADJ_SCHEMA = {
  type: 'object',
  properties: {
    away: { type: 'string' }, home: { type: 'string' },
    spread_adjustments: { type: 'array', items: { type: 'object', properties: {
      category: { type: 'string' }, points: { type: 'number' },
      evidence: { type: 'string' }, source: { type: 'string' } },
      required: ['category', 'points', 'evidence', 'source'] } },
    total_adjustments: { type: 'array', items: { type: 'object', properties: {
      category: { type: 'string' }, points: { type: 'number' },
      evidence: { type: 'string' }, source: { type: 'string' } },
      required: ['category', 'points', 'evidence', 'source'] } },
    tt_modifiers: { type: 'array', items: { type: 'object', properties: {
      team: { type: 'string' }, points: { type: 'number' }, reason: { type: 'string' } },
      required: ['team', 'points', 'reason'] } },
    considered_but_zero: { type: 'array', items: { type: 'object', properties: {
      item: { type: 'string' }, why_zero: { type: 'string' } },
      required: ['item', 'why_zero'] } },
    origin_note: { type: 'string' },
    tt_split_note: { type: 'string' },
    brief: { type: 'string' },
    flags: { type: 'array', items: { type: 'string' } } },
  required: ['away', 'home', 'spread_adjustments', 'total_adjustments', 'tt_modifiers',
             'considered_but_zero', 'origin_note', 'tt_split_note', 'brief', 'flags'],
}

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: {
      list: { type: 'string', enum: ['spread', 'total', 'tt'] },
      index: { type: 'number' },
      verdict: { type: 'string', enum: ['APPROVE', 'REJECT', 'CORRECT'] },
      corrected_points: { type: 'number' },
      reason: { type: 'string' } },
      required: ['list', 'index', 'verdict', 'reason'] } },
    missing_adjustments: { type: 'array', items: { type: 'object', properties: {
      list: { type: 'string', enum: ['spread', 'total', 'tt'] },
      category: { type: 'string' }, points: { type: 'number' }, team: { type: 'string' },
      evidence: { type: 'string' }, source: { type: 'string' } },
      required: ['list', 'category', 'points', 'evidence', 'source'] } },
    brief_ok: { type: 'boolean' },
    brief_problems: { type: 'string' },
    notes: { type: 'string' } },
  required: ['verdicts', 'missing_adjustments', 'brief_ok', 'notes'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    silent_injuries: { type: 'array', items: { type: 'object', properties: {
      game: { type: 'string' }, team: { type: 'string' }, player: { type: 'string' },
      why_it_matters: { type: 'string' } },
      required: ['game', 'team', 'player', 'why_it_matters'] } },
    other_gaps: { type: 'array', items: { type: 'string' } },
    ok_to_publish: { type: 'boolean' },
    notes: { type: 'string' } },
  required: ['silent_injuries', 'other_gaps', 'ok_to_publish', 'notes'],
}

phase('Adjust')

const analyzed = await pipeline(
  args.games,
  (g) => agent(`${RULES}

ROLE: ANALYST for ${g.away} @ ${g.home} only. First, load your game's evidence bundle by running this exact Bash command:
${READ_CMD(g.away, g.home)}

Then originate §5 context adjustments, §6 team-total modifiers, and the published game brief, working ONLY from that bundle. Every non-zero adjustment needs a NAMED evidence item and a source string from the bundle; "feels high/low" is forbidden. An empty adjustment list is a good answer when the engines already price everything. Anything you weighed and zeroed goes in considered_but_zero (include every premium-position absence from the bundle's injuries section — QB, OT, WR1, EDGE, CB1 — either as an adjustment or a considered_but_zero entry; that is spec rule 7, "no silent injuries"). Also write:
- origin_note: 1-3 sentences, cold and audit-ready: what moved the number and by how much.
- brief: one paragraph (5-8 sentences) for the published card covering (a) the Tier-A read — PFF point-spread ratings for both teams and the resulting spread_pff at HFA 1.75 (from cores.pff_meta), PFF QB ratings, plus nfelo's spread and QB context (use nfelo_baked_in and qb_context); (b) where Kevin Cole sits vs nfelo and PFF — quote all three spreads, the blend (cores.spread_core), and the disagreement (spread_sd); name which engine is the outlier when one is; then one clause on the TPT diagnostic (Donchess/FF-Winners gap vs the core, weight 0); (c) the total: quote the three implied (derived) totals and the core, and name the research components that set it (Week-1 prior 45.0, DIV, ENV, EFF from totals_basis.components) — no engine publishes a total; (d) adjustments applied with magnitudes under the Week-1 caps (±2.0/±2.5), or that none were, and what happened to the v5 baseline items (kept / re-sized / dropped); (e) the team-total split from the identity (no reallocations in v6). Cite source names (PFF, nfelo, Kevin Cole, Donchess...). No hype, numbers first.
- tt_split_note: one sentence on the identity split (T/2 ± S/2); tt_modifiers must be [].
- flags: data issues for Appendix C (TPT panel diagnostic-only/blank systems, unconfirmed QB, unverifiable roster codes, no weather forecast for outdoor game, structural clamp or sleeve clamp fired in cores).
Echo away='${g.away}', home='${g.home}' in your output.`,
    { label: `adj:${g.away}@${g.home}`, phase: 'Adjust', schema: ADJ_SCHEMA }),

  (proposal, g) => proposal && agent(`${RULES}

ROLE: ADVERSARIAL AUDITOR for ${g.away} @ ${g.home}. Load the same evidence bundle by running this exact Bash command:
${READ_CMD(g.away, g.home)}

An analyst proposed the adjustments below. Your default stance is REJECT: an adjustment survives only if it (a) names real evidence present in the bundle, (b) sits inside its §5 category range, (c) does not double-count something already inside the nfelo number (QB starter changes, rest, travel, HFA — check nfelo_baked_in and qb_context: e.g. if nfelo already carries a big negative QB adj for a team's downgraded starter, a further qb_change needs NEW post-8/31 news), (d) does not anchor toward the market, and (e) is applied to the correct object (spread vs team total vs game total — a WR1 absence hits the team total, not automatically the spread), and (f) is not one of the v6-deleted categories: weather (no verified forecasts exist), rest_short_week, travel_international, pace/style, divisional or dome/outdoor effects (all priced in the cores or deleted by the panel) are automatically REJECT. A qb_change must follow the stint table (2.5/1.5/0.5, net of what nfelo already baked; planned offseason starters 0) — CORRECT it to the table value otherwise. A02/"decode unverified" roster codes support at most a small flagged tweak, not max-range points. §6: ANY non-empty tt_modifiers entry is REJECT (v6 is identity-only; matchup reallocations were overturned by the panel). Confirm the brief quotes v6 numbers (HFA 1.75 PFF/Cole spreads, the v6 implied totals and components) and not the v5 ones. Verify every number and factual claim in the brief against the bundle (brief_ok=false with specifics if wrong). If the analyst MISSED a mandatory item (rule 7: a premium-position absence in the bundle appearing in neither adjustments nor considered_but_zero), add it to missing_adjustments with evidence — or as a 0-point entry with the zero-reason if that is the right call. Give a reason for every verdict; use CORRECT with corrected_points when direction is right but size is out of range.

ANALYST PROPOSAL:
${JSON.stringify(proposal, null, 1)}`,
    { label: `audit:${g.away}@${g.home}`, phase: 'Verify', schema: AUDIT_SCHEMA })
      .then(audit => ({ game: { away: g.away, home: g.home }, proposal, audit }))
)

const kept = analyzed.filter(Boolean)
log(`Adjust+Verify complete for ${kept.length}/16 games`)

phase('Critic')
const critic = await agent(`${RULES}

ROLE: COMPLETENESS CRITIC for the whole 16-game Week 1 card. First load all injury/QB evidence by running:
python3 -c "import json;d=json.load(open('/home/user/originator-2026-w01/bundles.json'));print(json.dumps([{'away':g['away'],'home':g['home'],'injuries':g['injuries'],'qb':{'away':(g['qb_context']['away'] or {}).get('starting_qb'),'home':(g['qb_context']['home'] or {}).get('starting_qb')}} for g in d['games']],indent=1))"

Then enforce spec rule 7 — "No silent injuries" — against the outcomes below: every QB, OT, WR1, EDGE, CB1 listed OUT/IR/PUP/SUSPENDED/QUESTIONABLE in the bundles must appear in an applied adjustment, a considered_but_zero entry, or the auditor's missing_adjustments for its game. List every violation in silent_injuries. Also list other_gaps: games whose flags omit a visible data issue (TPT systems blank, unconfirmed QB, outdoor game with no forecast, a structural/sleeve clamp that fired), games where a v6-deleted category (rest, travel, pace, weather, matchup reallocation) survived, or where a qb_change is not on the stint table. Set ok_to_publish=false ONLY for rule-7 violations or factual contradictions, not for already-flagged data scarcity.

OUTCOMES (per game: analyst adjustments + zeroed items + auditor additions + flags):
${JSON.stringify(kept.map(k => ({ game: k.game, spread_adjustments: k.proposal.spread_adjustments, total_adjustments: k.proposal.total_adjustments, tt_modifiers: k.proposal.tt_modifiers, considered_but_zero: k.proposal.considered_but_zero, auditor_missing: k.audit && k.audit.missing_adjustments, auditor_verdicts: k.audit && k.audit.verdicts, flags: k.proposal.flags })), null, 1)}`,
  { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA })

return { games: kept, critic }
