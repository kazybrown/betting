#!/usr/bin/env python3
"""Deterministic editor remediations for the v6 pass (auditor brief_problems +
critic other_gaps). Every change is recorded in adjustments.json under
editor_remediations_v6 for the audit trail."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
A = json.loads((RUN / "adjustments.json").read_text())
B = json.loads((RUN / "briefs.json").read_text())
C = {(g["away"], g["home"]): g for g in json.loads((RUN / "cores.json").read_text())}
ag = {(g["away"], g["home"]): g for g in A["games"]}
bg = {(b["away"], b["home"]): b for b in B["games"]}
log = []
W = {"nfelo": .46, "pff": .39, "cole": .15}


def sub(kind, key, old, new, field="brief"):
    obj = bg[key] if kind == "brief" else ag[key]
    if kind == "flags":
        hits = [i for i, f in enumerate(obj["flags"]) if old in f]
        assert hits, (key, old[:60])
        obj["flags"][hits[0]] = obj["flags"][hits[0]].replace(old, new)
    else:
        assert old in obj[field], (key, field, old[:60])
        obj[field] = obj[field].replace(old, new)
    log.append({"game": f"{key[0]}@{key[1]}", "where": f"{kind}.{field}" if kind != "flags" else "flags",
                "old": old[:160], "new": new[:160]})


RULING = (" ENGINE RULING (v6 editor remediation, cross-card): the §1 single-computer clamp limits the sleeve occupant's pull on the blend "
          "(|spread_core − Tier-A blend| ≤ 1.0), exactly as the v5 engine applied it; Cole's raw distance from the Tier-A blend is not clamped. "
          "Pull here {pull:+.2f} (raw distance {dist:+.2f}): no clamp fired.")


def ruling(key):
    g = C[key]
    ta = (W["nfelo"] * g["spread_nfelo"] + W["pff"] * g["spread_pff"]) / (W["nfelo"] + W["pff"])
    return RULING.format(pull=g["spread_core"] - ta, dist=g["spread_cole"] - ta)

# ---- SF@LA (auditor: two factual claims; minor v5 numerals; Cole diag basis) + clamp ruling
k = ("SF", "LA")
sub("brief", k, "built from the Week-1 prior 45.0, DIV -0.85 (divisional), ENV +2.0 on the listed dome roof and rating terms of +2.2 to +2.7 (the nfelo path also carries GT_dev 7.35 x 0.30 = +2.2 and a -0.05 QB term).",
    "built from the Week-1 prior 45.0, DIV -0.85 (divisional), ENV +2.0 on the listed dome roof and rating terms PFF +2.70 / Cole +2.22 / nfelo +0.48, the nfelo path adding 0.30 x GT_dev 7.35 = +2.21 and 0.72 x QB_sum -0.075 = -0.05.")
sub("brief", k, "the TPT diagnostic (weight 0) averages Donchess -4.3 and FF-Winners +2.4 to -2.96, +0.48 off the core, with PIR/STJ blank.",
    "the TPT panel value (weight 0) is -2.96 (Donchess -4.3 / FF-Winners +2.4, weighted inside the TPT file, not a simple mean), +0.48 off the core, with PIR/STJ blank.")
sub("brief", k, "and no sleeve clamp is reported;", "and the Cole sleeve's pull on the blend (-0.23) is inside the 1.0 §1 clamp;")
sub("flags", k, "v5 numbers (spread_pff -3.707, spread_cole -3.819, spread_core -2.939, total_core 48.488) are superseded and not quoted on the card.",
    "v5 numbers are superseded (different HFA and totals formula) and are not quoted on the card.")
sub("flags", k, "Cole's betting-PR diagnostic (-1.619) sits 3.1 points from his power-rating spread (-4.75);",
    "Cole's betting-PR diagnostic (-1.619, on the v5 nfelo neutral-HFA basis; -2.55 on the v6 0.75 basis) sits 3.1 points from his power-rating spread (-4.75) on the v5 basis and 2.2 on the v6 basis;")
sub("flags", k, "so the 1.0 sleeve clamp did not fire under the engine's definition - confirm the clamp's reference point.",
    "so the 1.0 sleeve clamp did not fire under the engine's definition." + ruling(k))
for c in ag[k]["considered_but_zero"]:
    if c["item"].startswith("Kevin Cole betting-PR diagnostic"):
        c["why_zero"] += " Basis note (editor): the -1.619 diagnostic embeds the v5 nfelo neutral HFA; on the v6 0.75 basis it is -2.55 and the internal gap is 2.2, not 3.1."
        log.append({"game": "SF@LA", "where": "considered_but_zero", "old": "gap 3.1", "new": "basis note added (2.2 on v6 basis)"})

# ---- BUF@HOU (auditor: deleted-categories misstatement, GT_dev/QB contributions; critic: relabel WR item)
k = ("BUF", "HOU")
sub("brief", k, "and the v5 rest, travel, weather and motivation zeros stay zero with their categories deleted by the panel.",
    "and rest and travel (categories deleted by the panel), weather (no verified forecast, so no numeric term; ENV already carries the roof) and motivation (Weeks 15-18 only) stay at zero.")
sub("brief", k, "(the nfelo path also GT_dev -0.13 and QB sum +0.15)", "(the nfelo path also carries 0.30 x GT_dev -0.13 = -0.04 and 0.72 x QB_sum +0.147 = +0.11)")
sub("adj", k, "the v5 rest, travel, weather and motivation entries were already zero and their categories are deleted.",
    "the v5 rest and travel entries stay zero with their categories deleted; weather (no verified forecast) and motivation (Weeks 15-18 only) stay zero.", field="origin_note")
item = ag[k]["total_adjustments"][0]
assert item["category"] == "qb_change" and "Higgins" in item["evidence"]
item["category"] = "skill_absence_wr"
item["clause"] = ("§5 category-1 skill-absence sub-clause (WR), not a stint-table QB change; eligibility: the IR predates 8/31, but a WR2/3 absence is not "
                  "inside the season-long team ratings (auditor APPROVED at the -0.5 floor); relabelled by the editor on the critic's finding")
log.append({"game": "BUF@HOU", "where": "total_adjustments[0].category", "old": "qb_change", "new": "skill_absence_wr (clause noted)"})

# ---- NYJ@TEN (auditor: v5 numerals; EFF gloss)
k = ("NYJ", "TEN")
sub("brief", k, "its v5 reference numbers (-3.155 / 42.482) are retired for the v6 cores", "its v5 reference numbers are retired for the v6 cores")
sub("brief", k, "EFF -0.13 (3.0 x prior-season EPA/play deviation -0.042, both offenses bottom-3 in 2025)",
    "EFF -0.13 (3.0 x epa_sum_dev -0.042, both teams' prior-season EPA/play efficiency vs league; separately, both offenses were bottom-3 in 2025)")

# ---- sleeve-clamp wording (critic cross-card gap)
k = ("ATL", "PIT")
sub("brief", k, "1.82 from the renormalized nfelo/PFF blend of -3.471, beyond the 1.0 sleeve clamp that cores did not record as fired)",
    "1.82 from the renormalized nfelo/PFF blend of -3.471; the §1 sleeve clamp limits Cole's pull on the blend, +0.27 here, so it did not fire)")
sub("flags", k, "Cole betting-PR diagnostic -3.697 is closer to PFF and was not used", "Cole betting-PR diagnostic -3.697 is closer to PFF and was not used." + ruling(k))
k = ("DEN", "KC")
sub("brief", k, "and Cole's 1.60 distance from the Tier-A blend (-1.146) is flagged against the §1 sleeve clamp of 1.0 because cores reports it as not fired.",
    "and Cole's 1.60 distance from the Tier-A blend (-1.146) is inside the §1 sleeve clamp as the engine applies it (the clamp limits the sleeve's pull on the blend, -0.24 here), so no clamp fired.")
sub("flags", k, "Engine to confirm the clamp basis; if the clamp applies at 1.0 the core would print near -1.30. Not altered by the analyst.",
    "Not altered by the analyst." + ruling(k))
k = ("NO", "DET")
sub("flags", k, "engine-side verification recommended", "engine-side verification done:" + ruling(k))
k = ("NE", "SEA")
sub("flags", k, "No structural clamp (PFF vs nfelo, 4.5) and no Cole sleeve clamp (1.0) fired: structural_flag null, sleeve_clamp_note null",
    "No structural clamp (PFF vs nfelo, 4.5) and no Cole sleeve clamp (1.0) fired: structural_flag null, sleeve_clamp_note null." + ruling(k))

# ---- WAS@PHI (critic: reconcile the auditor's precondition wording)
k = ("WAS", "PHI")
item = ag[k]["total_adjustments"][0]
item["editor_reconciliation"] = ("v6 editor: the auditor's note that the 'injured/weak OT' precondition is not met at the player level and the analyst's 'weakly met' "
                                 "reading are reconciled as a unit-level reading of 'weak OT' (PFF OL 32nd vs PHI DL 4th; no injured WAS OT); the item stays at the "
                                 "-1.0 floor on that reading, which the auditor APPROVED. Same unit-level reading as CHI@CAR and CLE@JAX (one leg inferred, held at or below mid-range).")
ag[k]["audit_notes"] += " [editor] Precondition wording reconciled: unit-level 'weak OL' reading, item held at the floor — see total_adjustments[0].editor_reconciliation."
log.append({"game": "WAS@PHI", "where": "total_adjustments[0]", "old": "precondition 'NOT met' vs APPROVE", "new": "editor_reconciliation recorded (unit-level reading, floor)"})

# ---- ledger completeness (critic)
ag[("DEN", "KC")]["considered_but_zero"].append({
    "item": "DEN non-premium reserves: interior OL Michael Deiter RES/R36, depth DL Matt Henningsen RES/R36, LB Levelle Bailey RES/R34; P Jeremy Crawshaw E14 (International Player Pathway exemption)",
    "why_zero": "Non-premium positions (not QB/OT/WR1/EDGE/CB1) and reserve-code evidence only; not priced; carried as a flag. Ledger entry added by the editor for cross-card consistency."})
ag[("GB", "MIN")]["considered_but_zero"].append({
    "item": "MIN P Brett Thorson E14 (International Player Pathway exemption)",
    "why_zero": "Roster exemption, not an injury; non-premium position; ledger entry only (editor)."})
log += [{"game": "DEN@KC", "where": "considered_but_zero", "old": "", "new": "non-premium reserves entry added"},
        {"game": "GB@MIN", "where": "considered_but_zero", "old": "", "new": "Thorson E14 entry added"}]

A["editor_remediations_v6"] = log
(RUN / "adjustments.json").write_text(json.dumps(A, indent=1))
(RUN / "briefs.json").write_text(json.dumps(B, indent=1))
print(f"{len(log)} remediations applied")
