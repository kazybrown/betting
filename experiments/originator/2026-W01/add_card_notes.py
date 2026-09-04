#!/usr/bin/env python3
"""Re-attach the card-wide Appendix C notes to adjustments.json after a full
reconcile run (reconcile.py rewrites the file without them)."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
NOTES_V6 = [
    "BUILD v6 (2026-09-04): three engines only — nfelo site model spreads, PFF point-spread ratings, Kevin Cole (Unexpected Points) power ratings as of 9/1/26 — at spread weights .46/.39/.15; the Prediction Tracker panel (Donchess/FF-Winners) is DIAGNOSTIC ONLY (weight 0) because five of its seven required computers are blank. The v6 parameters are the ones the 10-expert adversarial research panel upheld (research_config.json, research/synthesis.json): HFA 1.75 on the PFF/Cole paths; totals on the panel formula (Week-1 prior 45.0 + rating term + DIV + ENV, blended .38/.32/.30, plus the EPA efficiency term); Week-1 §5 caps ±2.0/±2.5; team totals by identity only (matchup reallocations overturned); rest/short-week/bye/west-east clauses deleted; QB stint table 2.5/1.5/0.5 net of inputs; confidence tags from the distance to the latest market line (diagnostic).",
    "TOTALS: no engine in this build publishes a game total. Every total is a derived implied total from net ratings (PFF-, nfelo-, Cole-implied on the v6 formula); their mutual agreement is partly structural (shared prior, DIV and ENV terms), so the total tag measures distance to the market, not source independence.",
    "Roster status code A02 appears under two labels — 'QUESTIONABLE (decode unverified)' and 'PUP-type (per an unofficial code map)' — for the same code. Read every A02 as 'Active/PUP-type, decode unverified'; official game-week injury reports (not yet published) supersede all roster codes.",
    "Cross-bundle QB2 inconsistency: Quinn Ewers appears as JAX QB2 (acquired from MIA ~Aug 30) and still among MIA backups in the MIA@LV roster note; immaterial to any number (QB1s unaffected), recorded as a feed-freshness issue.",
    "nfelo publishes model spreads and win probabilities only; the nfelo spreads on this card are the nfeloapp.com values supplied 2026-09-02 (confirmed unchanged 2026-09-04; 7 of 16 differ from the 09-01 repo snapshot).",
    "No verified weather forecast exists for any Week 1 game; outdoor games carry the panel's league-average outdoor ENV term (-0.5) only, and no per-game wind/temperature adjustment was made.",
]
p = RUN / "adjustments.json"
A = json.loads(p.read_text())
A["card_notes"] = NOTES_V6
p.write_text(json.dumps(A, indent=1))
print(f"card_notes ({len(NOTES_V6)}) attached to adjustments.json")
