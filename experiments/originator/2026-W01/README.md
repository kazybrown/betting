# ORIGINATOR — NFL 2026 Week 1 origin card

Run date: 2026-09-01 · Data status: **DEGRADED** (see Appendix C in the card)

Weekly origination of projected spreads, game totals, and team totals per the
ORIGINATOR methodology: PFF + nfelo Tier-A engines, light TPT ensemble sleeve,
evidence-named context adjustments, identity-reconciled team totals. Market
lines are reference only (Appendix B), never an input.

## Deliverables

- `output/week_01_origin_card.md` — the published card (slate table, per-game
  briefs, source matrix, market delta, data issues)
- `output/week_01_origin_card.csv` — machine-readable card
- `output/week_01_audit.json` — every input, weight, adjustment, clamp, and
  the full recovery sweep with sources

## Build (v5, 2026-09-04)

Per the user's instruction, the card is built from **three engines only**:
nfelo (site model spreads), PFF (point-spread ratings) and **Kevin Cole**
(Unexpected Points power rankings as of 9/1/26, read from the subscriber
workbook via the Google Drive connector; `data/raw/2026-W01/cole_power_rankings.csv`).
Cole occupies the Tier-B sleeve slot (spread .15 / total .30, single-source
clamp); nfelo .46 / PFF .39 keep Tier A. The Prediction Tracker panel is
carried as a **diagnostic only** (weight 0). No engine publishes a game total,
so every total is a §3.2-derived implied total from net ratings (nfelo-, PFF-
and Cole-implied, blended .32/.38/.30) and total confidence is forced LOW.

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

`league_total_prior = 46.0` (2025 REG realized mean, 46.03).

## Pipeline (reproducible)

```
python3 originator_engine.py --stage cores     # Tier-A priors + TPT sleeves + blend (§3-§4)
python3 build_bundles.py                       # per-game evidence bundles
# multi-agent pass: per-game §5 analysts -> adversarial auditors -> completeness critic
python3 reconcile.py <workflow_output>         # verdict application -> adjustments.json/briefs.json
python3 originator_engine.py --stage final     # §5 caps, §6 team totals, §7 confidence, §8 rounding
python3 originator_engine.py --stage publish --generated <ISO> --data-status DEGRADED
```

Intermediate artifacts checked in: `sweep.json` (recovery results),
`cores.json`, `bundles.json`, `adjustments.json` (post-audit), `briefs.json`,
`final.json`. The `*_v1.json` files are the pre-PFF-table pass (rank-model
PFF spreads), retained for the audit trail; the v2 adjust/verify pass used
the audited v1 adjustment set as its baseline.
