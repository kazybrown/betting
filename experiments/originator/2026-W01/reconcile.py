#!/usr/bin/env python3
"""Reconcile the adjust/verify workflow output into adjustments.json + briefs.json.

Policy (deterministic):
- Auditor verdict REJECT drops the adjustment; CORRECT replaces its points with
  corrected_points; APPROVE (or no verdict on an item) keeps it.
- Auditor missing_adjustments are added, marked added_by_auditor (the critic
  pass reviewed these too).
- Weather adjustments are force-dropped (no verified forecasts exist).
- Engine-side §5 sum caps still apply downstream.
Usage: reconcile.py <workflow_output_json>
"""

import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent


def main(path):
    blob = json.loads(Path(path).read_text())
    res = blob["result"] if "result" in blob else blob
    out_games, briefs, problems = [], [], []

    for entry in res["games"]:
        g = entry["game"]
        prop = entry["proposal"]
        audit = entry.get("audit") or {}
        verdict_map = {}
        for v in audit.get("verdicts", []):
            verdict_map[(v["list"], int(v["index"]))] = v

        def filter_list(name, items):
            kept = []
            for i, adj in enumerate(items):
                v = verdict_map.get((name, i))
                if v and v["verdict"] == "REJECT":
                    problems.append(f"{g['away']}@{g['home']}: dropped {name}[{i}] "
                                    f"{adj.get('category', adj.get('reason', '?'))} "
                                    f"({adj['points']:+.2f}) — {v['reason'][:120]}")
                    continue
                if v and v["verdict"] == "CORRECT" and v.get("corrected_points") is not None:
                    problems.append(f"{g['away']}@{g['home']}: corrected {name}[{i}] "
                                    f"{adj['points']:+.2f} -> {v['corrected_points']:+.2f} — "
                                    f"{v['reason'][:120]}")
                    adj = dict(adj, points=v["corrected_points"], audited="corrected")
                if adj.get("category", "").lower().startswith("weather") and adj.get("points"):
                    problems.append(f"{g['away']}@{g['home']}: force-dropped weather "
                                    f"adjustment ({adj['points']:+.2f}) — no verified forecast")
                    continue
                kept.append(adj)
            return kept

        s_adj = filter_list("spread", prop.get("spread_adjustments", []))
        t_adj = filter_list("total", prop.get("total_adjustments", []))
        tt_mod = filter_list("tt", prop.get("tt_modifiers", []))

        for miss in audit.get("missing_adjustments", []):
            if not miss.get("points"):
                continue  # zero-point documentation entries stay in the audit trail only
            item = {"category": miss.get("category", "auditor_added"),
                    "points": miss["points"], "evidence": miss.get("evidence", ""),
                    "source": miss.get("source", ""), "added_by_auditor": True}
            if miss["list"] == "spread":
                s_adj.append(item)
            elif miss["list"] == "total":
                t_adj.append(item)
            elif miss["list"] == "tt" and miss.get("team"):
                tt_mod.append({"team": miss["team"], "points": miss["points"],
                               "reason": miss.get("evidence", ""), "added_by_auditor": True})

        out_games.append({
            "away": g["away"], "home": g["home"],
            "spread_adjustments": s_adj, "total_adjustments": t_adj,
            "tt_modifiers": tt_mod,
            "considered_but_zero": prop.get("considered_but_zero", []),
            "origin_note": prop.get("origin_note", ""),
            "tt_split_note": prop.get("tt_split_note", ""),
            "flags": prop.get("flags", []),
            "audit_notes": audit.get("notes", ""),
        })
        briefs.append({"away": g["away"], "home": g["home"],
                       "brief": prop.get("brief", ""),
                       "brief_ok": audit.get("brief_ok", True),
                       "brief_problems": audit.get("brief_problems", "")})

    critic = res.get("critic") or {}
    if "--merge" in sys.argv:
        A = json.loads((RUN / "adjustments.json").read_text()); B = json.loads((RUN / "briefs.json").read_text())
        upd = {(g["away"], g["home"]): g for g in out_games}; ub = {(b["away"], b["home"]): b for b in briefs}
        A["games"] = [upd.get((g["away"], g["home"]), g) for g in A["games"]]
        B["games"] = [ub.get((b["away"], b["home"]), b) for b in B["games"]]
        A.setdefault("critic_partial_reruns", []).append({"games": [f"{a}@{h}" for a, h in upd], "critic": critic})
        (RUN / "adjustments.json").write_text(json.dumps(A, indent=1)); (RUN / "briefs.json").write_text(json.dumps(B, indent=1))
        print(f"MERGED {len(upd)} game(s) into adjustments.json/briefs.json")
    else:
        (RUN / "adjustments.json").write_text(json.dumps(
            {"games": out_games, "critic": critic}, indent=1))
        (RUN / "briefs.json").write_text(json.dumps({"games": briefs}, indent=1))

    print(f"adjustments.json + briefs.json written for {len(out_games)} games")
    print(f"\n--- audit interventions ({len(problems)}) ---")
    for p in problems:
        print(" ", p)
    bad_briefs = [b for b in briefs if not b["brief_ok"]]
    print(f"\n--- briefs flagged by auditors: {len(bad_briefs)} ---")
    for b in bad_briefs:
        print(f"  {b['away']}@{b['home']}: {b['brief_problems'][:200]}")
    print("\n--- critic ---")
    print(" ok_to_publish:", critic.get("ok_to_publish"))
    for si in critic.get("silent_injuries", []):
        print("  SILENT INJURY:", si)
    for gap in critic.get("other_gaps", []):
        print("  gap:", gap)


if __name__ == "__main__":
    main(sys.argv[1])
