#!/usr/bin/env python3
"""ORIGINATOR engine — NFL 2026 Week 1 origination per the ORIGINATOR spec.

Deterministic math for §3 (rating->points), §4 (blend), §5 (adjustment caps),
§6 (team totals), §7 (uncertainty), §8 (rounding), §9 (outputs).

Stages:
  --stage cores  : build Tier-A priors + TPT panels + unadjusted cores -> cores.json
  --stage final  : apply adjustments.json (validated per-game adjustments and
                   team-total modifiers), produce output/ card, csv, audit json.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
RAW = RUN / "data" / "raw" / "2026-W01"
OUT = RUN / "output"

SEASON, WEEK = 2026, 1
LEAGUE_TOTAL_PRIOR = 46.0  # 2025 REG realized mean total (46.03), per §3.2 update rule
ELO_PER_POINT = 25.0

TEAM_FIX = {
    "LAR": "LA", "OAK": "LV", "JAC": "JAX", "WSH": "WAS", "SD": "LAC", "STL": "LA",
    "BLT": "BAL", "HST": "HOU", "CLV": "CLE", "ARZ": "ARI",  # PFF abbreviations
}
VALID = set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split())

NAME_TO_ABBR = {
    "arizona": "ARI", "cardinals": "ARI", "atlanta": "ATL", "falcons": "ATL",
    "baltimore": "BAL", "ravens": "BAL", "buffalo": "BUF", "bills": "BUF",
    "carolina": "CAR", "panthers": "CAR", "chicago": "CHI", "bears": "CHI",
    "cincinnati": "CIN", "bengals": "CIN", "cleveland": "CLE", "browns": "CLE",
    "dallas": "DAL", "cowboys": "DAL", "denver": "DEN", "broncos": "DEN",
    "detroit": "DET", "lions": "DET", "green bay": "GB", "packers": "GB",
    "houston": "HOU", "texans": "HOU", "indianapolis": "IND", "colts": "IND",
    "jacksonville": "JAX", "jaguars": "JAX", "kansas city": "KC", "chiefs": "KC",
    "rams": "LA", "la rams": "LA", "l.a. rams": "LA", "los angeles rams": "LA",
    "chargers": "LAC", "los angeles chargers": "LAC", "la chargers": "LAC",
    "las vegas": "LV", "raiders": "LV", "miami": "MIA", "dolphins": "MIA",
    "minnesota": "MIN", "vikings": "MIN", "new england": "NE", "patriots": "NE",
    "new orleans": "NO", "saints": "NO", "giants": "NYG", "n.y. giants": "NYG",
    "new york giants": "NYG", "jets": "NYJ", "n.y. jets": "NYJ", "new york jets": "NYJ",
    "philadelphia": "PHI", "eagles": "PHI", "pittsburgh": "PIT", "steelers": "PIT",
    "seattle": "SEA", "seahawks": "SEA", "san francisco": "SF", "49ers": "SF",
    "tampa bay": "TB", "buccaneers": "TB", "tennessee": "TEN", "titans": "TEN",
    "washington": "WAS", "commanders": "WAS",
}


def norm_team(t):
    if not t:
        return None
    t = t.strip()
    if t.upper() in VALID:
        return t.upper()
    if t.upper() in TEAM_FIX:
        return TEAM_FIX[t.upper()]
    return NAME_TO_ABBR.get(t.lower())


def round_half(x):
    """§8: half-up to .0/.5 — x.00-.24 -> .0, x.25-.74 -> .5, x.75-.99 -> +1.0.
    Works on signed values by rounding the magnitude (so -3.25 -> -3.5)."""
    s = -1.0 if x < 0 else 1.0
    m = abs(x)
    base = math.floor(m)
    frac = m - base
    if frac < 0.25:
        r = base
    elif frac < 0.75:
        r = base + 0.5
    else:
        r = base + 1.0
    return 0.0 if r == 0 else s * r


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    tot = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2.0:
            return v
    return pairs[-1][0]


def rank_to_z(rank, n=32):
    """Normal-score for a 1..n rank (1 = best) -> z, best ~ +1.86."""
    if rank is None:
        return None
    p = (rank - 0.5) / n
    # inverse normal CDF (Acklam approximation is overkill; use statistics.NormalDist)
    return -statistics.NormalDist().inv_cdf(p)


def load_slate():
    rows = list(csv.DictReader(open(RAW / "nflverse_games.csv")))
    slate = []
    for r in rows:
        if r["season"] == str(SEASON) and r["week"] == str(WEEK) and r["game_type"] == "REG":
            home, away = norm_team(r["home_team"]), norm_team(r["away_team"])
            slate.append({
                "game_id": r["game_id"], "away": away, "home": home,
                "gameday": r["gameday"], "weekday": r["weekday"], "gametime": r["gametime"],
                "venue": r["stadium"], "roof": r["roof"] or "retractable",
                "surface": r["surface"], "location": r["location"],
                "away_rest": int(r["away_rest"]), "home_rest": int(r["home_rest"]),
                "div_game": r["div_game"] == "1",
                # nfldata convention: spread_line positive = home favored.
                # ORIGINATOR convention: negative = home favored -> flip sign.
                "market_spread": -float(r["spread_line"]) if r["spread_line"] else None,
                "market_total": float(r["total_line"]) if r["total_line"] else None,
                "away_ml": r["away_moneyline"], "home_ml": r["home_moneyline"],
            })
    assert len(slate) == 16, f"expected 16 games, got {len(slate)}"
    return slate


def load_nfelo():
    """spread_nfelo from nfelo's published projected home spread (already
    negative-favored convention), plus elo snapshot pts_vs_avg and per-game mods."""
    spreads = {}
    for r in csv.DictReader(open(RAW / "prediction_tracker.csv")):
        h, a = norm_team(r["home_team"]), norm_team(r["away_team"])
        spreads[(a, h)] = {
            "spread_nfelo": float(r["nfelo_projected_home_spread"]),
            "home_wp": float(r["nfelo_projected_home_win_probability"]),
            "spread_nfelo_repo_snapshot": float(r["nfelo_projected_home_spread"]),
            "home_wp_repo_snapshot": float(r["nfelo_projected_home_win_probability"]),
            "nfelo_source": "greerreNFL/nfelo prediction_tracker.csv (repo snapshot)",
        }
    # §12: nfelo site spreads pasted by the user are authoritative for the run.
    # Win probability: kept from the snapshot where the spread is unchanged,
    # otherwise interpolated on nfelo's own published (line -> WP) pairs.
    site = RAW / "nfelo_site_week1_pasted.csv"
    if site.exists():
        pairs = {}
        for v in spreads.values():
            pairs.setdefault(v["spread_nfelo_repo_snapshot"], []).append(v["home_wp_repo_snapshot"])
        xs = sorted(pairs)
        ys = [statistics.mean(pairs[x]) for x in xs]

        def interp(x):
            if x <= xs[0]:
                i = 0
            elif x >= xs[-1]:
                i = len(xs) - 2
            else:
                i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        for r in csv.DictReader(open(site)):
            key = (norm_team(r["away"]), norm_team(r["home"]))
            if key not in spreads:
                continue
            new = float(r["home_spread"])
            e = spreads[key]
            if abs(new - e["spread_nfelo_repo_snapshot"]) > 1e-9:
                e["home_wp"] = round(interp(new), 3)
                e["home_wp_basis"] = "interpolated on nfelo's published line->WP pairs (site spread differs from snapshot)"
            else:
                e["home_wp_basis"] = "nfelo published WP (snapshot; spread unchanged on site)"
            e["spread_nfelo"] = new
            e["nfelo_site_delta"] = round(new - e["spread_nfelo_repo_snapshot"], 2)
            e["nfelo_source"] = "nfeloapp.com Week 1 model spreads pasted by the user 2026-09-02 (authoritative per §12)"
    pts = {}
    for r in csv.DictReader(open(RAW / "elo_snapshot.csv")):
        pts[norm_team(r["team"])] = float(r["pts_vs_avg"])
    hdr = open(RAW / "nfelo_games_header.csv").read().strip().split(",")
    mods = {}
    for row in csv.reader(open(RAW / "nfelo_games_2026w1.csv")):
        d = dict(zip(hdr, row))
        gid = d["game_id"]  # 2026_01_AWAY_HOME
        parts = gid.split("_")
        a, h = norm_team(parts[2]), norm_team(parts[3])
        mods[(a, h)] = {
            "hfa_mod_elo": float(d["hfa_mod"]),
            "time_adv_elo": float(d["home_time_advantage_mod"]),
            "home_qb_adj_elo": float(d["home_538_qb_adj"]),
            "away_qb_adj_elo": float(d["away_538_qb_adj"]),
            "nfelo_line_close": float(d["nfelo_home_line_close"]),
            "market_line_close_nfelo": float(d["home_line_close"]) if d.get("home_line_close") else None,
        }
    return spreads, pts, mods


def load_pff_ratings():
    """PFF Power Rankings table (pasted by the user 2026-09-01 — authoritative
    for this run per §12). Point Spread Rating = points vs league average, QB
    component included (column sums to ~0)."""
    p = RAW / "pff_power_ratings.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        tm = norm_team(r["Team"])
        if tm:
            out[tm] = {
                "psr": float(r["Point Spread Rating Points"]),
                "psr_qb": float(r["Point Spread Rating QB"]),
                "proj_wins": float(r["Projections Avg. Wins"]),
                "proj_playoffs_pct": float(r["Projections Make Playoffs"]),
                "sos_remaining_rank": int(r["Strength of Schedule Remaining"]),
            }
    return out


def load_dvoa():
    dv = {}
    for r in csv.DictReader(open(RAW / "dvoa_projections_2026.csv")):
        if r["season"] == str(SEASON):
            dv[norm_team(r["team"])] = float(r["projected_total_dvoa"])
    return dv


def zmap(d):
    """z-score a {team: value} dict across its values."""
    vals = [v for v in d.values() if v is not None]
    if len(vals) < 8:
        return {k: None for k in d}
    mu, sd = statistics.mean(vals), statistics.pstdev(vals)
    if sd == 0:
        return {k: 0.0 for k in d}
    return {k: (v - mu) / sd if v is not None else None for k, v in d.items()}


# ---------------------------------------------------------------- sweep intake

def load_sweep():
    p = RUN / "sweep.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pff_tables(sweep):
    """Build per-team PFF-basis tables from the sweep. Returns (units, power_rank,
    explicit_spreads) — any may be sparse/empty."""
    units, power, explicit = {}, {}, {}
    pw = sweep.get("pffPower") or {}
    for t in pw.get("teams", []):
        tm = norm_team(t.get("team"))
        if tm and t.get("rank") is not None:
            power[tm] = {"rank": t["rank"], "verbatim": t.get("verbatim", False),
                         "source": t.get("source", "")}
    for g in pw.get("week1_spreads", []):
        a, h = norm_team(g.get("away")), norm_team(g.get("home"))
        blob = (str(g.get("source", "")) + " " + str(g.get("quote", ""))).upper()
        # rule 1 guard: recovered "spreads" that the sweep itself labeled as
        # market lines (NOT PFF) must never enter the origin blend
        if "NOT PFF" in blob or "MARKET LINE" in blob:
            continue
        if a and h and g.get("home_spread") is not None:
            explicit[(a, h)] = {"home_spread": g["home_spread"], "quote": g.get("quote", ""),
                                "source": g.get("source", "")}
    pu = sweep.get("pffUnits") or {}
    for t in pu.get("teams", []):
        tm = norm_team(t.get("team"))
        if tm:
            units[tm] = t
    return units, power, explicit


TPT_SPREAD_W = {"DONC": 0.34, "FFW": 0.24, "PIR": 0.22, "STJ": 0.20}
TPT_TOTAL_W = {"RPXL": 0.24, "RWP": 0.22, "DOK": 0.20, "DONC": 0.18, "FFW": 0.16}


def tpt_panels(sweep):
    """Per-game recovered TPT system values -> {(a,h): {system: value}} for
    spreads and totals. Derives spread/total from predicted scores when needed."""
    spreads, totals = {}, {}
    for section, target in (("tptSpreads", spreads), ("tptTotals", totals)):
        blob = sweep.get(section) or {}
        for v in blob.get("values", []):
            a, h = norm_team(v.get("away")), norm_team(v.get("home"))
            sysname = (v.get("system") or "").upper()
            if not a or not h or not sysname:
                continue
            key = (a, h)
            hs, tt = v.get("home_spread"), v.get("total")
            ph, pa = v.get("pred_home_score"), v.get("pred_away_score")
            if ph is not None and pa is not None:
                if hs is None:
                    hs = -(ph - pa)
                if tt is None:
                    tt = ph + pa
            if section == "tptSpreads" and hs is not None:
                target.setdefault(key, {})[sysname] = {"v": float(hs), "src": v.get("source", ""), "q": v.get("quote", "")}
            if section == "tptTotals" and tt is not None:
                target.setdefault(key, {})[sysname] = {"v": float(tt), "src": v.get("source", ""), "q": v.get("quote", "")}
    return spreads, totals


def tpt_blend(game_systems, default_w):
    """Weighted median over recovered systems with spec default weights
    renormalized over present systems (no YTD error data pre-W1 -> no shrink step,
    documented). Returns (value, used_systems) or (None, [])."""
    if not game_systems:
        return None, []
    present = {s: d for s, d in game_systems.items() if s in default_w}
    if not present:
        return None, []
    w = {s: default_w[s] for s in present}
    tot = sum(w.values())
    w = {s: x / tot for s, x in w.items()}
    # cap any one system at 0.32 of sleeve equivalent -> with few systems this
    # cap is unreachable; apply anyway for fidelity when >=4 present
    if len(w) >= 4:
        capped = {s: min(x, 0.32) for s, x in w.items()}
        tot = sum(capped.values())
        w = {s: x / tot for s, x in capped.items()}
    vals = [present[s]["v"] for s in w]
    wm = weighted_median(vals, [w[s] for s in w])
    # §2 step 5: shrink 40% toward the unweighted median so one system cannot dominate
    return 0.6 * wm + 0.4 * statistics.median(vals), sorted(w)


# ---------------------------------------------------------------- core builder

def build_cores():
    slate = load_slate()
    nf_spreads, nf_pts, nf_mods = load_nfelo()
    dvoa = load_dvoa()
    pffr = load_pff_ratings()
    sweep = load_sweep()
    units, power, explicit_pff = pff_tables(sweep)
    tpt_s, tpt_t = tpt_panels(sweep)
    tpt_mkt = {(norm_team(m["away"]), norm_team(m["home"])): m
               for m in (sweep.get("tptMarket") or {}).get("games", [])}

    dvoa_z = zmap(dvoa)
    power_z = {t: rank_to_z(d["rank"]) for t, d in power.items()}

    # unit z tables from ranks (1..32) where present
    def unit_z(field):
        vals = {t: (u.get(field) if isinstance(u.get(field), (int, float)) else None)
                for t, u in units.items()}
        return {t: rank_to_z(v) if v is not None else None for t, v in vals.items()}

    ol_z = unit_z("ol_rank")
    rush_z = unit_z("pass_rush_rank")
    cov_z = unit_z("coverage_rank")
    qb_z = unit_z("qb_rank")
    offg = zmap({t: u.get("off_grade") for t, u in units.items()})
    defg = zmap({t: u.get("def_grade") for t, u in units.items()})

    games = []
    for g in slate:
        a, h = g["away"], g["home"]
        key = (a, h)
        nf = nf_spreads.get(key)
        mods = nf_mods.get(key, {})
        spread_nfelo = nf["spread_nfelo"] if nf else None
        home_wp = nf["home_wp"] if nf else None
        nfelo_prov = {k: nf.get(k) for k in ("spread_nfelo_repo_snapshot", "nfelo_site_delta",
                                              "home_wp_basis", "nfelo_source")} if nf else {}

        hfa_pts = None
        if mods:
            hfa_pts = (mods["hfa_mod_elo"] + mods["time_adv_elo"]) / ELO_PER_POINT

        # ---- PFF spread path
        pff_meta = {"basis": None}
        unit_diag = None
        if a in pffr and h in pffr:
            # §3.1: PFF publishes an explicit point-spread rating -> use it;
            # the unit/rank model is kept only as a diagnostic (computed below)
            hfa = hfa_pts if hfa_pts is not None else 1.6
            spread_pff = -(pffr[h]["psr"] - pffr[a]["psr"]) - hfa
            pff_meta = {"basis": "explicit PFF point-spread rating (pasted table, authoritative per §12)",
                        "psr_home": pffr[h]["psr"], "psr_away": pffr[a]["psr"],
                        "psr_qb_home": pffr[h]["psr_qb"], "psr_qb_away": pffr[a]["psr_qb"],
                        "hfa_pts": round(hfa, 2)}
        elif key in explicit_pff:
            spread_pff = explicit_pff[key]["home_spread"]
            pff_meta = {"basis": "explicit PFF Week 1 spread",
                        "source": explicit_pff[key]["source"], "quote": explicit_pff[key]["quote"]}
        else:
            # unit-grade model §3.1 from rank-derived z-scores; ST edge unavailable -> 0
            oh, oa = offg.get(h), offg.get(a)
            dh, da = defg.get(h), defg.get(a)
            # fall back to power-rank z when unit off/def grades absent
            if oh is None and h in power_z:
                oh = power_z.get(h)
            if oa is None and a in power_z:
                oa = power_z.get(a)
            if dh is None and h in power_z:
                dh = power_z.get(h)
            if da is None and a in power_z:
                da = power_z.get(a)
            olh, ola = ol_z.get(h), ol_z.get(a)
            rsh, rsa = rush_z.get(h), rush_z.get(a)
            if None not in (oh, oa, dh, da):
                off_edge = (oh - da) / math.sqrt(2)
                def_edge = (dh - oa) / math.sqrt(2)
                ol_edge = 0.0
                if None not in (olh, ola, rsh, rsa):
                    ol_edge = ((olh - rsa) - (ola - rsh)) / math.sqrt(2)
                st_edge = 0.0
                hfa = hfa_pts if hfa_pts is not None else 1.6
                spread_pff = -(2.4 * off_edge + 2.1 * def_edge + 0.4 * st_edge + 0.6 * ol_edge) - hfa
                pff_meta = {"basis": "unit/rank z-model (§3.1), ST edge unavailable",
                            "off_edge": round(off_edge, 3), "def_edge": round(def_edge, 3),
                            "ol_edge": round(ol_edge, 3), "hfa_pts": round(hfa, 2)}
            else:
                spread_pff = None
                pff_meta = {"basis": "PFF unavailable for this game"}
        if pff_meta.get("psr_home") is not None:
            # unit/rank z-model diagnostic alongside the explicit rating
            oh, oa = power_z.get(h), power_z.get(a)
            if oh is not None and oa is not None:
                hfa = pff_meta["hfa_pts"]
                unit_diag = round(-(4.5 / math.sqrt(2)) * (oh - oa) - hfa, 3)
            pff_meta["rank_model_diag"] = unit_diag

        # §3.1: clamp any single PFF-vs-nfelo disagreement above 4.5 points and
        # note it as a structural flag (usually QB or OL)
        structural_flag = None
        if spread_pff is not None and spread_nfelo is not None:
            diff = spread_pff - spread_nfelo
            if abs(diff) > 4.5:
                clamped = spread_nfelo + math.copysign(4.5, diff)
                structural_flag = (f"PFF/nfelo disagreement {diff:+.2f} pts exceeds 4.5; "
                                   f"spread_pff clamped {spread_pff:+.2f} -> {clamped:+.2f} "
                                   "(structural — usually QB or OL)")
                spread_pff = clamped

        # ---- totals
        pace_adj = 0.0  # populated in final stage from validated adjustments
        # nfelo-implied total (§3.2 fallback: 0.35 pts of total per combined rating point)
        total_nfelo = None
        if a in nf_pts and h in nf_pts:
            total_nfelo = LEAGUE_TOTAL_PRIOR + 0.35 * (nf_pts[a] + nf_pts[h])

        # PFF-implied total: requires PFF off/def quality (§3.2). The DVOA
        # projection z is kept as a DIAGNOSTIC only — §4's missing-source
        # protocol governs the blend when PFF is absent; Tier-A substitution
        # is not authorized by the spec.
        total_pff = None
        tot_basis = None
        ozh, oza = offg.get(h), offg.get(a)
        dzh, dza = defg.get(h), defg.get(a)
        if None not in (ozh, oza, dzh, dza):
            total_pff = (LEAGUE_TOTAL_PRIOR + 2.8 * (ozh + oza) / 2.0
                         - 2.4 * (dzh + dza) / 2.0)
            tot_basis = "PFF off/def grade z (§3.2)"
        total_dvoa_diag = None
        if dvoa_z.get(a) is not None and dvoa_z.get(h) is not None:
            total_dvoa_diag = round(LEAGUE_TOTAL_PRIOR + 0.4 * (dvoa_z[a] + dvoa_z[h]), 3)

        # ---- TPT sleeves
        spread_tpt, s_sys = tpt_blend(tpt_s.get(key, {}), TPT_SPREAD_W)
        total_tpt, t_sys = tpt_blend(tpt_t.get(key, {}), TPT_TOTAL_W)

        games.append({
            **g,
            "spread_nfelo": spread_nfelo, "home_wp_nfelo": home_wp, "nfelo_provenance": nfelo_prov,
            "spread_pff": None if spread_pff is None else round(spread_pff, 3),
            "pff_meta": pff_meta,
            "spread_tpt": spread_tpt, "spread_tpt_systems": s_sys,
            "tpt_spread_detail": {s: d["v"] for s, d in tpt_s.get(key, {}).items()},
            "total_nfelo": None if total_nfelo is None else round(total_nfelo, 3),
            "total_pff": None if total_pff is None else round(total_pff, 3),
            "total_pff_basis": tot_basis,
            "total_dvoa_diag": total_dvoa_diag,
            "dvoa_proj": {"home": dvoa.get(h), "away": dvoa.get(a)},
            "total_tpt": total_tpt, "total_tpt_systems": t_sys,
            "tpt_total_detail": {s: d["v"] for s, d in tpt_t.get(key, {}).items()},
            "structural_flag": structural_flag,
            "market_open_spread": tpt_mkt.get(key, {}).get("open_spread"),
            "market_open_total": tpt_mkt.get(key, {}).get("open_total"),
            "market_tpt_spread": tpt_mkt.get(key, {}).get("current_spread"),
            "market_tpt_total": tpt_mkt.get(key, {}).get("current_total"),
            "pff_psr": {"home": pffr.get(h, {}).get("psr"), "away": pffr.get(a, {}).get("psr"),
                        "home_qb": pffr.get(h, {}).get("psr_qb"), "away_qb": pffr.get(a, {}).get("psr_qb"),
                        "home_proj_wins": pffr.get(h, {}).get("proj_wins"),
                        "away_proj_wins": pffr.get(a, {}).get("proj_wins")},
            "hfa_pts_nfelo": None if hfa_pts is None else round(hfa_pts, 3),
            "nfelo_qb_adj": {"home_elo": mods.get("home_qb_adj_elo"),
                              "away_elo": mods.get("away_qb_adj_elo")},
            "nfelo_vs_market_close": (
                None if not mods or mods.get("market_line_close_nfelo") is None
                else round(mods["nfelo_line_close"] - mods["market_line_close_nfelo"], 2)),
        })

    # ---- blend §4 with per-game missing-source protocol
    for g in games:
        sn, sp, st = g["spread_nfelo"], g["spread_pff"], g["spread_tpt"]
        tn, tp, tt = g["total_nfelo"], g["total_pff"], g["total_tpt"]

        if sp is not None and st is not None:
            g["spread_core"] = 0.46 * sn + 0.39 * sp + 0.15 * st
            g["spread_weights"] = {"nfelo": 0.46, "pff": 0.39, "tpt": 0.15}
            n_sys = len(g["spread_tpt_systems"])
            if n_sys <= 2:  # §1 single-computer clamp, 1.0/system on spreads
                w1, w2 = 0.46 / 0.85, 0.39 / 0.85
                tier_a = w1 * sn + w2 * sp
                limit = 1.0 * n_sys
                drift = g["spread_core"] - tier_a
                if abs(drift) > limit:
                    g["spread_core"] = tier_a + math.copysign(limit, drift)
                    g["tpt_spread_clamped"] = round(drift, 2)
        elif sp is not None:  # missing TPT -> renormalize Tier A
            w1, w2 = 0.46 / 0.85, 0.39 / 0.85
            g["spread_core"] = w1 * sn + w2 * sp
            g["spread_weights"] = {"nfelo": round(w1, 3), "pff": round(w2, 3), "tpt": 0.0}
        elif st is not None:  # missing PFF
            g["spread_core"] = 0.70 * sn + 0.30 * st
            g["spread_weights"] = {"nfelo": 0.70, "pff": 0.0, "tpt": 0.30}
            n_sys = len(g["spread_tpt_systems"])
            if n_sys <= 2:  # §1 single-computer clamp vs Tier A (nfelo alone)
                limit = 1.0 * n_sys
                drift = g["spread_core"] - sn
                if abs(drift) > limit:
                    g["spread_core"] = sn + math.copysign(limit, drift)
                    g["tpt_spread_clamped"] = round(drift, 2)
        else:
            g["spread_core"] = sn
            g["spread_weights"] = {"nfelo": 1.0, "pff": 0.0, "tpt": 0.0}

        if tp is not None and tt is not None:
            g["total_core"] = 0.38 * tp + 0.32 * tn + 0.30 * tt
            g["total_weights"] = {"pff": 0.38, "nfelo": 0.32, "tpt": 0.30}
            # §1 single-computer clamp for a degraded sleeve (<=2 systems):
            # the sleeve may not move the number more than 1.5/system vs Tier A
            n_sys = len(g["total_tpt_systems"])
            if n_sys <= 2:
                w1, w2 = 0.38 / 0.70, 0.32 / 0.70
                tier_a = w1 * tp + w2 * tn
                limit = 1.5 * n_sys
                drift = g["total_core"] - tier_a
                if abs(drift) > limit:
                    g["total_core"] = tier_a + math.copysign(limit, drift)
                    g["tpt_total_clamped"] = round(drift, 2)
        elif tp is not None:  # missing TPT
            w1, w2 = 0.38 / 0.70, 0.32 / 0.70
            g["total_core"] = w1 * tp + w2 * tn
            g["total_weights"] = {"pff": round(w1, 3), "nfelo": round(w2, 3), "tpt": 0.0}
        elif tt is not None:  # missing PFF
            g["total_core"] = 0.50 * tn + 0.50 * tt
            g["total_weights"] = {"pff": 0.0, "nfelo": 0.50, "tpt": 0.50}
            n_sys = len(g["total_tpt_systems"])
            if n_sys <= 2:  # §1 single-computer clamp vs Tier A (nfelo alone)
                limit = 1.5 * n_sys
                drift = g["total_core"] - tn
                if abs(drift) > limit:
                    g["total_core"] = tn + math.copysign(limit, drift)
                    g["tpt_total_clamped"] = round(drift, 2)
        else:
            g["total_core"] = tn
            g["total_weights"] = {"pff": 0.0, "nfelo": 1.0, "tpt": 0.0}

        # §1 Tier-B rule: no single TPT computer may move the origin number more
        # than 1.0 (spread) / 1.5 (total) vs the Tier-A blend. Each system's share
        # of the sleeve's effect E = core - tierA is E * (its renormalized sleeve
        # weight); if any share exceeds the limit, E is scaled down so the largest
        # share equals the limit (order-independent, monotone).
        for kind, detail_key, wdict, limit, core_key, tier, wkey in (
                ("spread", "tpt_spread_detail", TPT_SPREAD_W, 1.0, "spread_core", (sn, sp), "spread_weights"),
                ("total", "tpt_total_detail", TPT_TOTAL_W, 1.5, "total_core", (tn, tp), "total_weights")):
            detail = g[detail_key]
            wt = g[wkey]
            if not detail or wt["tpt"] == 0:
                continue
            n1, p1 = tier
            wa = wt["nfelo"] + wt["pff"]
            tier_a = (wt["nfelo"] * n1 + wt["pff"] * (p1 if p1 is not None else 0)) / wa
            E = g[core_key] - tier_a
            present = {k: wdict[k] for k in detail if k in wdict}
            tot = sum(present.values())
            shares = {k: E * v / tot for k, v in present.items()}
            worst = max(shares, key=lambda k: abs(shares[k]))
            if abs(shares[worst]) > limit:
                scale = limit / abs(shares[worst])
                g[core_key] = tier_a + E * scale
                g.setdefault("tpt_system_clamps", []).append(
                    {"kind": kind, "system": worst, "share": round(shares[worst], 2),
                     "limit": limit, "sleeve_effect_before": round(E, 2), "sleeve_effect_after": round(E * scale, 2)})

        g["spread_core"] = round(g["spread_core"], 3)
        g["total_core"] = round(g["total_core"], 3)

        # §7 uncertainty across available sources
        s_sources = [x for x in [sn, sp] if x is not None] + list(g["tpt_spread_detail"].values())
        t_sources = [x for x in [tn, tp] if x is not None] + list(g["tpt_total_detail"].values())
        g["spread_range"] = round(max(s_sources) - min(s_sources), 2) if len(s_sources) >= 2 else None
        g["total_range"] = round(max(t_sources) - min(t_sources), 2) if len(t_sources) >= 2 else None
        g["spread_sd"] = round(statistics.stdev(s_sources), 3) if len(s_sources) >= 2 else None
        g["total_sd"] = round(statistics.stdev(t_sources), 3) if len(t_sources) >= 2 else None
        g["n_spread_sources"] = len(s_sources)
        g["n_total_sources"] = len(t_sources)

    (RUN / "cores.json").write_text(json.dumps(games, indent=1))
    print(f"cores.json written: {len(games)} games")
    for g in games:
        print(f"  {g['away']}@{g['home']}: nfelo {g['spread_nfelo']}, pff {g['spread_pff']}, "
              f"tpt {g['spread_tpt']} -> core {g['spread_core']} | "
              f"totals nfelo {g['total_nfelo']}, pff {g['total_pff']}, tpt {g['total_tpt']} "
              f"-> core {g['total_core']}")
    return games


# ---------------------------------------------------------------- final stage

def conf_tag(sd, is_total, n_sources):
    if sd is None or n_sources < 3:
        return "LOW"
    if is_total:
        return "HIGH" if sd <= 1.8 else ("MED" if sd <= 3.0 else "LOW")
    return "HIGH" if sd <= 1.2 else ("MED" if sd <= 2.2 else "LOW")


def finalize():
    games = json.loads((RUN / "cores.json").read_text())
    adj_blob = json.loads((RUN / "adjustments.json").read_text())
    adj_by_key = {(a["away"], a["home"]): a for a in adj_blob["games"]}
    sweep = load_sweep()

    out_rows = []
    for g in games:
        key = (g["away"], g["home"])
        A = adj_by_key.get(key, {})
        s_adjs = A.get("spread_adjustments", [])
        t_adjs = A.get("total_adjustments", [])
        # §5 hard caps on the SUM of adjustments
        adj_spread = max(-2.5, min(2.5, sum(x["points"] for x in s_adjs)))
        adj_total = max(-3.0, min(3.0, sum(x["points"] for x in t_adjs)))

        spread_raw = g["spread_core"] + adj_spread
        total_raw = g["total_core"] + adj_total
        spread_pub = round_half(spread_raw)
        total_pub = round_half(total_raw)

        # §6 team totals from identity; S = home margin (negative = home favored)
        S, T = spread_raw, total_raw
        home_tt_raw = T / 2.0 - S / 2.0
        away_tt_raw = T / 2.0 + S / 2.0
        # matchup modifiers (reallocation; re-centered so sum returns to T)
        tt_mods = A.get("tt_modifiers", [])
        home_shift = sum(m["points"] for m in tt_mods if m["team"] == g["home"])
        away_shift = sum(m["points"] for m in tt_mods if m["team"] == g["away"])
        h1, a1 = home_tt_raw + home_shift, away_tt_raw + away_shift
        drift = (h1 + a1) - T
        h1, a1 = h1 - drift / 2.0, a1 - drift / 2.0

        # round team totals with identity protection vs published total
        h_pub, a_pub = round_half(h1), round_half(a1)
        if abs((h_pub + a_pub) - total_pub) > 1e-9:
            # keep the tt closer to raw, snap the other; if still broken by 0.5,
            # §0.6: adjust the TOTAL by 0.5 toward the unrounded sum
            if abs(h_pub - h1) <= abs(a_pub - a1):
                a_pub = total_pub - h_pub
            else:
                h_pub = total_pub - a_pub
            if min(a_pub, h_pub) < 0 or abs(a_pub * 2 - round(a_pub * 2)) > 1e-9:
                total_pub2 = total_pub + (0.5 if (h1 + a1) > total_pub else -0.5)
                total_pub = total_pub2
                a_pub = total_pub - h_pub

        g_out = dict(g)
        g_out.update({
            "adj_spread": round(adj_spread, 2), "adj_total": round(adj_total, 2),
            "spread_adjustment_log": s_adjs, "total_adjustment_log": t_adjs,
            "tt_modifier_log": tt_mods,
            "spread_raw": round(spread_raw, 3), "total_raw": round(total_raw, 3),
            "spread_origin": spread_pub, "total_origin": total_pub,
            "tt_home_raw": round(h1, 3), "tt_away_raw": round(a1, 3),
            "tt_home": h_pub, "tt_away": a_pub,
            "conf_spread": conf_tag(g["spread_sd"], False, g["n_spread_sources"]),
            "conf_total": conf_tag(g["total_sd"], True, g["n_total_sources"]),
            "origin_note": A.get("origin_note", ""),
            "tt_split_note": A.get("tt_split_note", ""),
            "flags": A.get("flags", []),
        })
        # §7 TPT-light override: if Tier A agree within 1 pt, floor confidence at MED
        if (g["spread_pff"] is not None and g["spread_nfelo"] is not None
                and abs(g["spread_pff"] - g["spread_nfelo"]) <= 1.0
                and g_out["conf_spread"] == "LOW"):
            g_out["conf_spread"] = "MED"
        # identity assertions
        assert abs((g_out["tt_home"] + g_out["tt_away"]) - g_out["total_origin"]) < 1e-9, g_out["game_id"]
        assert g_out["spread_origin"] % 0.5 == 0 and g_out["total_origin"] % 0.5 == 0
        out_rows.append(g_out)

    (RUN / "final.json").write_text(json.dumps(out_rows, indent=1))
    print("final.json written")
    for g in out_rows:
        print(f"  {g['away']}@{g['home']}: {g['home']} {g['spread_origin']:+.1f} | "
              f"T {g['total_origin']} | TT {g['tt_home']}/{g['tt_away']} | "
              f"adj {g['adj_spread']:+.1f}/{g['adj_total']:+.1f} | "
              f"{g['conf_spread']}/{g['conf_total']}")
    return out_rows


# ---------------------------------------------------------------- publish §9

CSV_COLS = ("season,week,game_id,away,home,kickoff,venue,roof,"
            "spread_origin,total_origin,tt_home,tt_away,home_wp,"
            "spread_nfelo,spread_pff,spread_tpt,spread_donc,spread_ffw,spread_pir,spread_stj,"
            "total_nfelo,total_pff,total_tpt,total_rpxl,total_ffw,total_rwp,total_donc,total_dok,"
            "spread_core,total_core,adj_spread,adj_total,conf_spread,conf_total,"
            "market_spread,market_total,market_tt_home,market_tt_away,"
            "data_status,notes").split(",")


def implied_tt(spread, total):
    """Market-implied team totals from market spread/total (home persp.)."""
    if spread is None or total is None:
        return None, None
    return round(total / 2.0 - spread / 2.0, 2), round(total / 2.0 + spread / 2.0, 2)


def publish(generated_iso, data_status):
    games = json.loads((RUN / "final.json").read_text())
    briefs = {}
    bp = RUN / "briefs.json"
    if bp.exists():
        briefs = {(b["away"], b["home"]): b for b in json.loads(bp.read_text())["games"]}
    OUT.mkdir(exist_ok=True)

    def fmt(x, nd=1):
        return "" if x is None else (f"{x:.{nd}f}")

    # ---------- CSV
    with open(OUT / "week_01_origin_card.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for g in games:
            sd, td = g["tpt_spread_detail"], g["tpt_total_detail"]
            note = "; ".join(g.get("flags", []))
            w.writerow({
                "season": SEASON, "week": WEEK, "game_id": g["game_id"],
                "away": g["away"], "home": g["home"],
                "kickoff": f"{g['gameday']} {g['gametime']}", "venue": g["venue"],
                "roof": g["roof"],
                "spread_origin": g["spread_origin"], "total_origin": g["total_origin"],
                "tt_home": g["tt_home"], "tt_away": g["tt_away"],
                "home_wp": g["home_wp_nfelo"],
                "spread_nfelo": g["spread_nfelo"], "spread_pff": fmt(g["spread_pff"], 2),
                "spread_tpt": fmt(g["spread_tpt"], 2),
                "spread_donc": fmt(sd.get("DONC"), 1), "spread_ffw": fmt(sd.get("FFW"), 1),
                "spread_pir": fmt(sd.get("PIR"), 1), "spread_stj": fmt(sd.get("STJ"), 1),
                "total_nfelo": fmt(g["total_nfelo"], 2), "total_pff": fmt(g["total_pff"], 2),
                "total_tpt": fmt(g["total_tpt"], 2),
                "total_rpxl": fmt(td.get("RPXL"), 1), "total_ffw": fmt(td.get("FFW"), 1),
                "total_rwp": fmt(td.get("RWP"), 1), "total_donc": fmt(td.get("DONC"), 1),
                "total_dok": fmt(td.get("DOK"), 1),
                "spread_core": g["spread_core"], "total_core": g["total_core"],
                "adj_spread": g["adj_spread"], "adj_total": g["adj_total"],
                "conf_spread": g["conf_spread"], "conf_total": g["conf_total"],
                "market_spread": g["market_spread"], "market_total": g["market_total"],
                "market_tt_home": implied_tt(g["market_spread"], g["market_total"])[0],
                "market_tt_away": implied_tt(g["market_spread"], g["market_total"])[1],
                "data_status": data_status, "notes": note,
            })

    # ---------- Markdown card
    L = []
    L.append(f"# NFL ORIGIN CARD — {SEASON} Week {WEEK}")
    L.append(f"Generated: {generated_iso}")
    L.append(f"Data status: {data_status}")
    wts = games[0]["spread_weights"]
    twts = games[0]["total_weights"]
    L.append(f"Weights (modal game): spread nfelo {wts['nfelo']} / PFF {wts['pff']} / TPT {wts['tpt']}"
             f" | total PFF {twts['pff']} / nfelo {twts['nfelo']} / TPT {twts['tpt']}"
             " (per-game missing-source renormalization applies; see Appendix C)")
    L.append("")
    L.append("## Slate table")
    L.append("")
    L.append("| Game | Kick | Origin Spread (home) | Origin Total | Home TT | Away TT | Home WP | Conf | Spread range | Total range |")
    L.append("|------|------|----------------------|--------------|---------|---------|---------|------|--------------|-------------|")
    for g in games:
        conf = f"{g['conf_spread'][0]}/{g['conf_total'][0]}"
        wp = f"{g['home_wp_nfelo']*100:.0f}%" if g["home_wp_nfelo"] else ""
        L.append(f"| {g['away']} @ {g['home']} | {g['weekday'][:3]} {g['gameday'][5:]} {g['gametime']} "
                 f"| {g['home']} {g['spread_origin']:+.1f} | {g['total_origin']:.1f} "
                 f"| {g['tt_home']:.1f} | {g['tt_away']:.1f} | {wp} | {conf} "
                 f"| {fmt(g['spread_range'])} | {fmt(g['total_range'])} |")
    L.append("")
    L.append("## Game briefs")
    L.append("")
    for g in games:
        b = briefs.get((g["away"], g["home"]), {})
        L.append(f"### {g['away']} @ {g['home']} — {g['home']} {g['spread_origin']:+.1f}, "
                 f"total {g['total_origin']:.1f} ({g['tt_home']:.1f}/{g['tt_away']:.1f})")
        L.append(b.get("brief", "_No brief available._"))
        L.append("")
    L.append("## Appendix A — Source matrix")
    L.append("")
    L.append("nfelo publishes model spreads and win probabilities only — no game total or projected score. "
             "Every 'nfelo-implied total' on this card is derived per §3.2 from nfelo's team ratings "
             "(league prior + 0.35 × combined points-vs-average) and is labelled as such; nfelo team totals do not exist.")
    L.append("")
    L.append("| Game | nfelo S | PFF S | DONC | FFW | PIR | STJ | RPXL | RWP | DOK | TPT med (S/T) | Market (S/T) |")
    L.append("|------|-------|-----|------|-----|-----|-----|------|-----|-----|---------------|--------------|")
    for g in games:
        sd, td = g["tpt_spread_detail"], g["tpt_total_detail"]
        L.append(f"| {g['away']}@{g['home']} | {g['spread_nfelo']:+.1f} | {fmt(g['spread_pff'],2) or '—'} "
                 f"| {fmt(sd.get('DONC')) or '—'} | {fmt(sd.get('FFW')) or '—'} | {fmt(sd.get('PIR')) or '—'} "
                 f"| {fmt(sd.get('STJ')) or '—'} | {fmt(td.get('RPXL')) or '—'} | {fmt(td.get('RWP')) or '—'} "
                 f"| {fmt(td.get('DOK')) or '—'} | {fmt(g['spread_tpt'],2) or '—'}/{fmt(g['total_tpt'],2) or '—'} "
                 f"| {g['market_spread']:+.1f}/{g['market_total']:.1f} |")
    L.append("")
    L.append("## Appendix B — Market delta (not an input)")
    L.append("")
    L.append("| Game | Origin S | Open S | Mkt S | ΔS vs mkt | Origin T | Open T | Mkt T | ΔT vs mkt | Origin TTh/TTa | Mkt TTh/TTa |")
    L.append("|------|----------|--------|-------|-----------|----------|--------|-------|-----------|----------------|-------------|")
    for g in games:
        mh, ma = implied_tt(g["market_spread"], g["market_total"])
        ds = g["spread_origin"] - g["market_spread"]
        dt = g["total_origin"] - g["market_total"]
        os_, ot = g.get("market_open_spread"), g.get("market_open_total")
        L.append(f"| {g['away']}@{g['home']} | {g['spread_origin']:+.1f} | {fmt(os_) if os_ is None else f'{os_:+.1f}'} | {g['market_spread']:+.1f} "
                 f"| {ds:+.1f} | {g['total_origin']:.1f} | {fmt(ot)} | {g['market_total']:.1f} | {dt:+.1f} "
                 f"| {g['tt_home']:.1f}/{g['tt_away']:.1f} | {mh}/{ma} |")
    L.append("")
    L.append("## Appendix C — Data issues")
    L.append("")
    seen = set()
    for g in games:
        for fl in g.get("flags", []):
            line = f"- {g['away']}@{g['home']}: {fl}"
            if line not in seen:
                seen.add(line)
                L.append(line)
    (OUT / "week_01_origin_card.md").write_text("\n".join(L) + "\n")

    # ---------- Audit JSON
    audit = {
        "season": SEASON, "week": WEEK, "generated": generated_iso,
        "data_status": data_status,
        "league_total_prior": LEAGUE_TOTAL_PRIOR,
        "league_total_prior_basis": "2025 REG realized mean total 46.03 (nflverse games.csv), per §3.2 update rule",
        "sources": {
            "nfelo": "greerreNFL/nfelo output_data (automated update committed 2026-09-01 15:40 PT; Week 1 projected spreads identical to the 2026-08-31 file, Elo snapshot refreshed): prediction_tracker.csv, elo_snapshot.csv, nfelo_games.csv",
            "schedule_market": "nflverse/nfldata games.csv (market snapshot for Appendix B only)",
            "pff": "PFF Power Rankings table (pff.com/betting/nfl-power-rankings) pasted by the user 2026-09-01 — authoritative per §12: Point Spread Rating (points vs avg, QB included) used directly per §3.1; earlier web-search rank recovery retained as diagnostic",
            "tpt": "The Prediction Tracker nflpredictions.csv + nfltotals.csv (Week 1) pasted by the user 2026-09-02 — authoritative per §12; Donchess and FF-Winners populated, Pi-Rate/Lou St. John/RP Excel/Laffaye/Dokter blank in TPT's file; opening lines from the same files feed Appendix B",
            "dvoa_diag": "greerreNFL/nfelo dvoa_projections.csv 2026 (diagnostic only, not blended)",
        },
        "conventions": {
            "spread": "home perspective, negative = home favored",
            "rounding": "half-up to .0/.5 per §8",
            "nfelo_total_derivation": "league_total_prior + 0.35*(home pts_vs_avg + away pts_vs_avg) per §3.2 fallback",
            "section5_team_total_adjustments": "§5 adjustments filed against a team total are applied to the game total and allocated by the identity (T/2 ± S/2); a one-sided allocation requires the category's linked spread leg or a §6 reallocation (auditor convention, v3)",
            "tpt_sleeve": "spec default weights renormalized over present systems (no inverse-MAE reweighting: no YTD error data pre-Week-1), weighted median then 40% shrink toward the unweighted median; single-computer clamp (1.0/1.5 pts per system) on sleeves with <=2 systems",
        },
        "games": games,
        "sweep": load_sweep(),
    }
    (OUT / "week_01_audit.json").write_text(json.dumps(audit, indent=1))
    print("published:", [p.name for p in sorted(OUT.iterdir())])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cores", "final", "publish"], required=True)
    ap.add_argument("--generated", default="")
    ap.add_argument("--data-status", default="DEGRADED")
    args = ap.parse_args()
    if args.stage == "cores":
        build_cores()
    elif args.stage == "final":
        finalize()
    else:
        publish(args.generated, args.data_status)
