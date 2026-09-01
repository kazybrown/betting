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

## Data provenance

- **nfelo** (Tier A2, fully live): `greerreNFL/nfelo` `output_data/` at the
  2026-08-31 automated-update commit — published Week 1 spreads/win probs,
  QB-adjusted Elo snapshot, per-game HFA/QB modifiers. Raw copies under
  `data/raw/2026-W01/`.
- **PFF** (Tier A1, partial): pff.com is blocked by the network egress
  allowlist; team power ranks (26/32 verbatim) and scattered unit ranks /
  2025-final team grades were recovered from web-search snippets with per-item
  citations (see `sweep.json`). Spread contribution built via the §3.1
  rank/unit z-model; PFF-implied totals unavailable.
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
`final.json`.
