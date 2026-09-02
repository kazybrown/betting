#!/usr/bin/env python3
"""Ingest a pasted Prediction Tracker table (prednfl.html rows or the
nflpredictions.csv / nfltotals.csv files) into sweep.json's TPT panels.

Usage:
  python3 ingest_tpt.py <file.csv> [--kind spreads|totals|auto] [--sign home-margin|origin]

Accepts any CSV/TSV whose columns include the home and road team plus one
column per prediction system. System columns are matched by substring:
  donc|donchess -> DONC   ffw|winners -> FFW   pirate|pi-rate -> PIR
  stjohn|st. john -> STJ  excel -> RPXL       rwp|laffaye -> RWP
  dokter|entropy -> DOK
Sign: TPT publishes predicted HOME MARGIN (positive = home favored). The
ORIGINATOR convention is negative = home favored, so values are negated by
default (--sign home-margin). Pass --sign origin if the pasted numbers are
already in the ORIGINATOR convention. A sanity table vs nfelo is printed —
confirm the sign before republishing.
"""
import csv, json, re, sys
from pathlib import Path
from originator_engine import norm_team, load_nfelo

RUN = Path(__file__).resolve().parent
SYS = [("donchess", "DONC"), ("donc", "DONC"), ("winners", "FFW"), ("ffw", "FFW"),
       ("pirate", "PIR"), ("pi-rate", "PIR"), ("stjohn", "STJ"), ("st. john", "STJ"),
       ("st john", "STJ"), ("excel", "RPXL"), ("laffaye", "RWP"), ("rwp", "RWP"),
       ("dokter", "DOK"), ("entropy", "DOK")]
SPREAD_SYS, TOTAL_SYS = {"DONC", "FFW", "PIR", "STJ"}, {"RPXL", "FFW", "RWP", "DONC", "DOK"}


def sys_code(col):
    c = col.lower()
    for k, v in SYS:
        if k in c:
            return v
    return None


def main():
    args = sys.argv[1:]
    path = args[0]
    kind = args[args.index("--kind") + 1] if "--kind" in args else "auto"
    sign = args[args.index("--sign") + 1] if "--sign" in args else "home-margin"
    text = Path(path).read_text()
    delim = "\t" if "\t" in text.splitlines()[0] else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    cols = rows[0].keys()
    home_col = next(c for c in cols if c.lower().strip() in ("home", "home team", "hometeam"))
    road_col = next(c for c in cols if c.lower().strip() in ("road", "away", "visitor", "road team", "away team"))
    if kind == "auto":
        kind = "totals" if any("tot" in c.lower() for c in cols) or "total" in path.lower() else "spreads"
    want = TOTAL_SYS if kind == "totals" else SPREAD_SYS
    sweep = json.loads((RUN / "sweep.json").read_text())
    key = "tptTotals" if kind == "totals" else "tptSpreads"
    blob = sweep.setdefault(key, {"values": [], "systems_missing": [], "sources": [], "notes": ""})
    if blob["values"]:
        blob["notes"] += f" | {len(blob['values'])} earlier web-recovered value(s) SUPERSEDED by the pasted TPT file (authoritative per §12)"
    blob["values"] = []
    blob["data_errors"] = []
    mkt = sweep.setdefault("tptMarket", {"games": [], "source": f"pasted TPT file {Path(path).name}"})
    nf, _, _ = load_nfelo()
    seen_sys, n = set(), 0
    print(f"kind={kind} sign={sign} | home col '{home_col}' road col '{road_col}'")
    for r in rows:
        h, a = norm_team(r[home_col]), norm_team(r[road_col])
        if not h or not a:
            print("  ! unmapped team:", r[home_col], r[road_col]); continue
        line = []
        for c in cols:
            code = sys_code(c)
            if code not in want:
                continue
            raw = (r[c] or "").strip()
            if raw in ("", "NA", "-", "null"):
                continue
            v = float(raw)
            if (kind == "totals" and not 30 <= v <= 65) or (kind == "spreads" and abs(v) > 28):
                blob["data_errors"].append({"away": a, "home": h, "system": code, "value": v, "column": c,
                                            "action": "dropped as data error (implausible magnitude); not replaced"})
                print(f"  ! {a}@{h} {code} {c}={raw} dropped as data error")
                continue
            if kind == "spreads":
                hs = -v if sign == "home-margin" else v
                blob["values"].append({"away": a, "home": h, "system": code, "home_spread": hs,
                                       "quote": f"{c}={raw}", "source": f"pasted TPT table {Path(path).name}"})
                line.append(f"{code} {hs:+.1f}")
            else:
                blob["values"].append({"away": a, "home": h, "system": code, "total": v,
                                       "quote": f"{c}={raw}", "source": f"pasted TPT table {Path(path).name}"})
                line.append(f"{code} {v:.1f}")
            seen_sys.add(code); n += 1
        lo, li = (r.get("lineopen") or "").strip(), (r.get("line") or "").strip()
        entry = next((m for m in mkt["games"] if m["away"] == a and m["home"] == h), None)
        if entry is None:
            entry = {"away": a, "home": h}; mkt["games"].append(entry)
        if kind == "spreads":
            if lo: entry["open_spread"] = -float(lo)
            if li: entry["current_spread"] = -float(li)
        else:
            if lo: entry["open_total"] = float(lo)
            if li: entry["current_total"] = float(li)
        nfs = nf.get((a, h), {}).get("spread_nfelo")
        print(f"  {a}@{h}: " + ", ".join(line) + (f"   | nfelo {nfs:+.1f}" if nfs is not None and kind == "spreads" else ""))
    blob["systems_missing"] = sorted(want - seen_sys)
    blob["sources"].append(f"pasted TPT table {Path(path).name} (authoritative per §12)")
    blob["notes"] += f" | {n} values ingested from pasted table; missing systems: {blob['systems_missing']}"
    (RUN / "sweep.json").write_text(json.dumps(sweep, indent=1))
    print(f"\n{n} values written to sweep.json[{key}]; systems missing: {blob['systems_missing']}")
    print("Next: python3 originator_engine.py --stage cores && python3 build_bundles.py  (then re-run adjust/verify)")


if __name__ == "__main__":
    main()
