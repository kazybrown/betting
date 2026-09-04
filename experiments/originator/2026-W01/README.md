# ORIGINATOR — NFL 2026 Week 1 origin card

Run date: 2026-09-04 (v6) · Data status: **DEGRADED** (see Appendix C in the card)

Weekly origination of projected spreads, game totals, and team totals per the
ORIGINATOR methodology: PFF + nfelo Tier-A engines, Kevin Cole in the sleeve
slot, evidence-named context adjustments, identity-reconciled team totals.
Market lines are reference only (Appendix B), never an input.

## Deliverables

- `output/week_01_origin_card.md` — the published card (slate table, method,
  per-game briefs, source matrix, market delta, data issues)
- `output/week_01_origin_card.csv` — machine-readable card
- `output/week_01_audit.json` — every input, weight, adjustment, clamp, and
  the full recovery sweep with sources
- `artifact_origin_card.html` — the published card page;
  `artifact_research_report.html` — the expert-panel research report

## Build v6 (2026-09-04) — research-calibrated

v6 applies the parameter changes the 10-expert adversarial research panel
upheld (`research/synthesis.json`, frozen in `research_config.json`) on top of
the v5 three-engine build. `research_engine.py` wraps the spec engine's
loaders: `--stage cores` runs the spec cores (kept as `cores_spec.json`) and
rebuilds spreads/totals with the research parameters; `--stage final` applies
the audited adjustments with the research caps and tags. What changed:

| Area | v5 (spec) | v6 (panel) |
|------|-----------|------------|
| HFA on PFF/Cole spread paths | nfelo per-game HFA (≈3.7) | 1.75 (0.75 international neutral, 0.0 domestic neutral) |
| Totals | 46.0 + 0.35 × rating sum, three paths | Week-1 prior 45.0 + rating term + DIV (−0.85/+0.45) + ENV (dome +2.0 / outdoor −0.5) [+ prior-season game-total and QB terms on the nfelo path], blended .38/.32/.30, plus EFF = clip(3.0 × EPA deviation, ±2.0) |
| §5 caps (Week 1) | ±2.5 / ±3.0 | ±2.0 / ±2.5 |
| §6 team totals | identity + audited matchup reallocations | identity only (reallocations overturned) |
| Rest / short week / bye / west-east | §5 clauses | deleted; rest priced once by the engine (Week 1 exempt) |
| QB change | −2.0..−4.5 | stint table 2.5 / 1.5 / 0.5 net of what the inputs embed; planned starters 0 |
| Confidence tags | source dispersion (totals forced LOW) | distance to the latest market line (diagnostic): spread <1.5/<3.0, total <2.5/<5.0 |
| Kept | 25 Elo/pt, nfelo QB term, structural clamp 4.5, blend weights, half-up rounding | unchanged |

The 16-game analyst → adversarial-auditor → completeness-critic pass was
re-run on the v6 cores under the v6 rules (`bundles.json` carries the rules
and the v5 audited set as the baseline to re-base). v5 is preserved as
`*_v5.json`; `final_research.json` is the earlier side-by-side comparison.

## Build v5 (2026-09-04)

Per the user's instruction, the card is built from **three engines only**:
nfelo (site model spreads), PFF (point-spread ratings) and **Kevin Cole**
(Unexpected Points power rankings as of 9/1/26, read from the subscriber
workbook via the Google Drive connector; `data/raw/2026-W01/cole_power_rankings.csv`).
Cole occupies the Tier-B sleeve slot (spread .15 / total .30, single-source
clamp); nfelo .46 / PFF .39 keep Tier A. The Prediction Tracker panel is
carried as a **diagnostic only** (weight 0). No engine publishes a game total,
so every total is a derived implied total from net ratings.

## Research panel

`research/` holds the panel: `kit.py` (loaders, fit ≤2021 / test 2022–25,
closing line as the benchmark), `data/` (nflverse games, nfelo history, Cole
team-game and QB tables, PFF ratings), `experts/<key>/` (scripts and logs for
the ten experts and their critics), `panel_results_raw.json` (61 theories:
35 upheld, 25 downgraded, 1 overturned) and `synthesis.json` (the nine
adopted changes). `make_report.py` renders the report page.

## Data provenance

- **nfelo** (Tier A2, fully live): `greerreNFL/nfelo` `output_data/` at the
  2026-08-31 automated-update commit — published Week 1 spreads/win probs,
  QB-adjusted Elo snapshot, per-game HFA/QB modifiers. Raw copies under
  `data/raw/2026-W01/`.
- **PFF** (Tier A1, clean for spreads): the user supplied PFF's Power
  Rankings table (pff.com/betting/nfl-power-rankings, `data/raw/2026-W01/
  pff_power_ratings.csv`) — authoritative per §12. Its Point Spread Rating
  (points vs league average, QB component included) drives `spread_pff`
  directly per §3.1: `-(PSR_home - PSR_away) - site HFA`. The earlier
  web-recovered post-FA power ranks (`sweep.json`) are kept as a diagnostic.
  PFF off/def unit grades were not available, so PFF-implied totals remain
  missing and the §4 missing-source protocol governs totals.
- **TPT panel** (Tier B, effectively missing): thepredictiontracker.com and
  all member-system sites blocked. One verbatim number recovered (Dokter
  Entropy 41.83 total, NE@SEA), applied under the single-computer clamp.
  Missing-source protocol (§4) governs all other games.
- **Schedule/market**: `nflverse/nfldata` `games.csv` fetched 2026-09-01
  (trimmed to 2025+ in `data/raw/`); market numbers appear only in Appendix B.
- **Injuries/QBs/quality**: nflverse roster/depth-chart feeds + local NFLkz
  2025 play-by-play (EPA/pace computed locally) + cited news snippets.

`league_total_prior = 46.0` (2025 REG realized mean, 46.03); v6 Week-1 prior 45.0 (−1.0 early-season offset).

## Pipeline (reproducible)

```
python3 research_engine.py --stage cores       # spec cores (cores_spec.json) -> v6 research cores (cores.json)
python3 build_bundles.py                       # per-game evidence bundles (v6 rules, v5 baseline)
# multi-agent pass: per-game §5 analysts -> adversarial auditors -> completeness critic
python3 reconcile.py <workflow_output>         # verdict application -> adjustments.json/briefs.json
python3 add_card_notes.py                      # card-wide Appendix C notes
python3 research_engine.py --stage final       # v6 caps, identity team totals, market-distance tags, §8 rounding
python3 originator_engine.py --stage publish --generated <ISO> --data-status DEGRADED
python3 make_artifact.py                       # HTML card page
```

Spec-default pipeline (v1–v5): `originator_engine.py --stage cores|final|publish`.

Intermediate artifacts checked in: `sweep.json` (recovery results),
`cores.json`, `bundles.json`, `adjustments.json` (post-audit), `briefs.json`,
`final.json`. The `*_v1.json` files are the pre-PFF-table pass (rank-model
PFF spreads), retained for the audit trail; the v2 adjust/verify pass used
the audited v1 adjustment set as its baseline.
