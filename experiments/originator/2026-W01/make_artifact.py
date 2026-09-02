#!/usr/bin/env python3
"""Render the Week 1 origin card as a self-contained HTML artifact page."""

import html
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
games = json.loads((RUN / "final.json").read_text())
briefs = {(b["away"], b["home"]): b for b in json.loads((RUN / "briefs.json").read_text())["games"]}
adjs = {(a["away"], a["home"]): a for a in json.loads((RUN / "adjustments.json").read_text())["games"]}
import subprocess
GEN = subprocess.run(["date","-u","+%Y-%m-%d %H:%M UTC"],capture_output=True,text=True).stdout.strip()


def esc(s):
    return html.escape(str(s))


def chip(tag):
    cls = {"HIGH": "ok", "MED": "mid", "LOW": "low"}[tag]
    return f'<span class="chip {cls}">{tag}</span>'


def f1(x, plus=False):
    if x is None:
        return "—"
    return f"{x:+.1f}" if plus else f"{x:.1f}"


def f2(x, plus=False):
    if x is None:
        return "—"
    return f"{x:+.2f}" if plus else f"{x:.2f}"


rows = []
for g in games:
    key = (g["away"], g["home"])
    ds = g["spread_origin"] - g["market_spread"]
    dt = g["total_origin"] - g["market_total"]
    kick = f"{g['weekday'][:3]} {g['gameday'][5:].replace('-', '/')}"
    rows.append(f"""<tr>
<td class="g"><a href="#b-{g['away']}-{g['home']}">{g['away']} @ {g['home']}</a></td>
<td class="mut">{kick}</td>
<td class="num strong">{g['home']} {f1(g['spread_origin'], plus=True)}</td>
<td class="num strong">{f1(g['total_origin'])}</td>
<td class="num">{f1(g['tt_home'])} / {f1(g['tt_away'])}</td>
<td class="num">{g['home_wp_nfelo']*100:.0f}%</td>
<td>{chip(g['conf_spread'])}{chip(g['conf_total'])}</td>
<td class="num mut">{f1(ds, plus=True)} / {f1(dt, plus=True)}</td>
</tr>""")

brief_blocks = []
for g in games:
    key = (g["away"], g["home"])
    b = briefs.get(key, {})
    a = adjs.get(key, {})
    adj_lines = ""
    logged = (g.get("spread_adjustment_log") or []) + (g.get("total_adjustment_log") or [])
    if logged or g.get("tt_modifier_log"):
        items = []
        for x in g.get("spread_adjustment_log", []):
            items.append(f"<li><span class=\"num\">{x['points']:+.2f}</span> spread — {esc(x['category'])}: {esc(x['evidence'][:220])}</li>")
        for x in g.get("total_adjustment_log", []):
            items.append(f"<li><span class=\"num\">{x['points']:+.2f}</span> total — {esc(x['category'])}: {esc(x['evidence'][:220])}</li>")
        for x in g.get("tt_modifier_log", []):
            items.append(f"<li><span class=\"num\">{x['points']:+.2f}</span> {esc(x['team'])} team total — {esc(x['reason'][:220])}</li>")
        adj_lines = "<h4>Adjustment log</h4><ul class=\"adj\">" + "".join(items) + "</ul>"
    else:
        adj_lines = "<p class=\"mut small\">No §5 adjustments or §6 modifiers cleared the evidence bar.</p>"
    flags = "".join(f"<li>{esc(fl)}</li>" for fl in g.get("flags", []))
    brief_blocks.append(f"""<details id="b-{g['away']}-{g['home']}">
<summary><span class="g">{g['away']} @ {g['home']}</span>
<span class="num strong">{g['home']} {f1(g['spread_origin'], plus=True)}</span>
<span class="num">T {f1(g['total_origin'])}</span>
<span class="num mut">TT {f1(g['tt_home'])}/{f1(g['tt_away'])}</span></summary>
<div class="body">
<p class="orig-note">{esc(g.get('origin_note',''))}</p>
<p>{esc(b.get('brief',''))}</p>
{adj_lines}
<h4>Data flags</h4><ul class="flags">{flags}</ul>
</div>
</details>""")

srcm = []
for g in games:
    sd, td = g["tpt_spread_detail"], g["tpt_total_detail"]
    srcm.append(f"""<tr><td class="g">{g['away']}@{g['home']}</td>
<td class="num">{f1(g['spread_nfelo'], plus=True)}</td><td class="num">{f2(g['spread_pff'], plus=True)}</td>
<td class="num mut">{f2(g['pff_psr']['home'], plus=True)} / {f2(g['pff_psr']['away'], plus=True)}</td>
<td class="num">{f2(g['total_nfelo'])}</td><td class="num">{f2(g['total_pff'])}</td>
<td class="num">{f1(sd.get('DONC'), plus=True)} / {f1(td.get('DONC'))}</td>
<td class="num">{f1(sd.get('FFW'), plus=True)} / {f1(td.get('FFW'))}</td>
<td class="num mut">{f1(g['market_spread'], plus=True)} / {f1(g['market_total'])}</td></tr>""")

mktd = []
for g in games:
    ds = g["spread_origin"] - g["market_spread"]
    dt = g["total_origin"] - g["market_total"]
    hl = ' class="hot"' if abs(ds) >= 1.5 or abs(dt) >= 2.0 else ""
    mktd.append(f"""<tr{hl}><td class="g">{g['away']}@{g['home']}</td>
<td class="num">{g['home']} {f1(g['spread_origin'], plus=True)}</td><td class="num mut">{f1(g['market_spread'], plus=True)}</td>
<td class="num strong">{f1(ds, plus=True)}</td>
<td class="num">{f1(g['total_origin'])}</td><td class="num mut">{f1(g['market_total'])}</td>
<td class="num strong">{f1(dt, plus=True)}</td></tr>""")

issues = []
seen = set()
for g in games:
    for fl in g.get("flags", []):
        line = f"<li><span class=\"g\">{g['away']}@{g['home']}</span> — {esc(fl)}</li>"
        if line not in seen:
            seen.add(line)
            issues.append(line)

page = f"""<title>Week 1 Origin Card</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root {{
  --ground:#F7F7F4; --surface:#FFFFFF; --ink:#1A2420; --mut:#5C6B64;
  --accent:#0E6B4F; --amber:#9A6A00; --rule:#D9DED9; --rowline:#E7EAE6;
  --chip-ok:#0E6B4F; --chip-ok-bg:#E3F0EA; --chip-mid:#7A5C00; --chip-mid-bg:#F4ECD4;
  --chip-low:#8C3A2B; --chip-low-bg:#F6E4DF; --hot:#FBF4E0;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#101614; --surface:#18201C; --ink:#E4EAE5; --mut:#8CA096;
    --accent:#4FC08D; --amber:#D9A62E; --rule:#2A3630; --rowline:#222C27;
    --chip-ok:#4FC08D; --chip-ok-bg:#173327; --chip-mid:#D9A62E; --chip-mid-bg:#33290F;
    --chip-low:#E08A6D; --chip-low-bg:#33110A; --hot:#241E0E;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#101614; --surface:#18201C; --ink:#E4EAE5; --mut:#8CA096;
  --accent:#4FC08D; --amber:#D9A62E; --rule:#2A3630; --rowline:#222C27;
  --chip-ok:#4FC08D; --chip-ok-bg:#173327; --chip-mid:#D9A62E; --chip-mid-bg:#33290F;
  --chip-low:#E08A6D; --chip-low-bg:#33110A; --hot:#241E0E;
}}
body {{ background:var(--ground); color:var(--ink);
  font:15px/1.55 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  margin:0; padding:0 20px 72px; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
header {{ padding:40px 0 20px; border-bottom:2px solid var(--ink); }}
.eyebrow {{ font:600 12px/1 "IBM Plex Mono", monospace; letter-spacing:.18em;
  text-transform:uppercase; color:var(--accent); margin:0 0 10px; }}
h1 {{ font:600 34px/1.1 "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  margin:0 0 8px; text-wrap:balance; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 22px; align-items:center;
  font:13px/1.4 "IBM Plex Mono", monospace; color:var(--mut); }}
.status {{ display:inline-block; font:600 11px/1 "IBM Plex Mono", monospace;
  letter-spacing:.12em; color:var(--amber); border:1px solid var(--amber);
  padding:4px 8px; border-radius:2px; }}
h2 {{ font:600 20px/1.2 "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  margin:44px 0 6px; letter-spacing:.02em; }}
h2 + p.lede {{ margin:0 0 14px; color:var(--mut); font-size:13.5px; max-width:68ch; }}
.tblwrap {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; min-width:820px; }}
th {{ font:600 11px/1.3 "IBM Plex Mono", monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--mut); text-align:left;
  padding:10px 12px; border-bottom:1px solid var(--ink); white-space:nowrap; }}
td {{ padding:9px 12px; border-bottom:1px solid var(--rowline); white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
tr.hot td {{ background:var(--hot); }}
.num {{ font-family:"IBM Plex Mono", monospace; font-variant-numeric:tabular-nums; }}
.strong {{ font-weight:600; color:var(--accent); }}
.mut {{ color:var(--mut); }}
.small {{ font-size:13px; }}
.g {{ font-weight:600; }}
.g a {{ color:inherit; text-decoration:none; border-bottom:1px dotted var(--mut); }}
.g a:hover, .g a:focus-visible {{ color:var(--accent); border-bottom-color:var(--accent); }}
a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.chip {{ display:inline-block; font:600 10px/1 "IBM Plex Mono", monospace;
  letter-spacing:.08em; padding:3px 6px; border-radius:2px; margin-right:4px; }}
.chip.ok  {{ color:var(--chip-ok);  background:var(--chip-ok-bg); }}
.chip.mid {{ color:var(--chip-mid); background:var(--chip-mid-bg); }}
.chip.low {{ color:var(--chip-low); background:var(--chip-low-bg); }}
details {{ border:1px solid var(--rule); background:var(--surface); margin:0 0 8px; }}
summary {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline;
  padding:11px 14px; cursor:pointer; list-style:none; }}
summary::-webkit-details-marker {{ display:none; }}
summary::before {{ content:"+"; font:600 14px/1 "IBM Plex Mono", monospace; color:var(--mut); }}
details[open] summary::before {{ content:"–"; }}
summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:-2px; }}
details .body {{ padding:2px 16px 16px 34px; max-width:75ch; }}
details .body p {{ margin:10px 0; }}
.orig-note {{ font-weight:600; }}
h4 {{ font:600 11px/1 "IBM Plex Mono", monospace; letter-spacing:.12em;
  text-transform:uppercase; color:var(--mut); margin:16px 0 6px; }}
ul.adj, ul.flags {{ margin:6px 0; padding-left:18px; }}
ul.adj li, ul.flags li {{ margin:4px 0; font-size:13.5px; }}
ul.flags li {{ color:var(--mut); }}
ul.issues {{ margin:10px 0; padding-left:18px; }}
ul.issues li {{ margin:6px 0; font-size:13.5px; color:var(--mut); }}
footer {{ margin-top:52px; padding-top:16px; border-top:1px solid var(--rule);
  font:12.5px/1.6 "IBM Plex Mono", monospace; color:var(--mut); }}
footer .num {{ color:var(--ink); }}
</style>
<div class="wrap">
<header>
<p class="eyebrow">Originated · not copied from market</p>
<h1>NFL Origin Card — 2026 Week&nbsp;1</h1>
<div class="meta">
<span class="status">DATA STATUS: DEGRADED</span>
<span>nfelo: site spreads (09-02) · PFF: point-spread ratings · TPT: Donchess + FF-Winners (5 systems blank) · weather: no forecasts · nfelo totals: derived, not published</span>
<span>Generated {GEN}</span>
<span>Spread wts (modal): nfelo .541 / PFF .459 / TPT 0</span>
<span>Total wts (modal): nfelo .50 / TPT .50 / PFF 0</span>
</div>
</header>

<h2>Slate</h2>
<p class="lede">Spread is home-team perspective, negative = home favored. Conf chips are spread / total. Δ mkt is reference only — the market is never an input to the origin number.</p>
<div class="tblwrap"><table>
<thead><tr><th>Game</th><th>Kick</th><th>Origin spread</th><th>Total</th><th>TT H / A</th><th>Home WP</th><th>Conf S·T</th><th>Δ mkt S / T</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>

<h2>Game briefs</h2>
<p class="lede">Tier-A read, panel status, adjustment log with named evidence, and the team-total split rationale for each game.</p>
{''.join(brief_blocks)}

<h2>Appendix A — Source matrix</h2>
<p class="lede">Per-game engine outputs; PSR = PFF point-spread rating (points vs league average, QB included). nfelo publishes model spreads and win probabilities only — no total or projected score — so the nfelo-implied total is derived per §3.2 from nfelo team ratings and is never an nfelo number. nfelo spreads are the nfeloapp.com site values supplied 2026-09-02. TPT panel systems (Donchess/DRatings, FF-Winners, Pi-Rate, Lou St. John, RP Excel, Laffaye RWP) were unrecoverable — the one recovered number is Dokter Entropy's NE@SEA total, applied under the single-computer clamp.</p>
<div class="tblwrap"><table>
<thead><tr><th>Game</th><th>nfelo S</th><th>PFF S</th><th>PSR H / A</th><th>nfelo-implied T (derived)</th><th>PFF T</th><th>DONC S / T</th><th>FFW S / T</th><th>Market S / T</th></tr></thead>
<tbody>{''.join(srcm)}</tbody>
</table></div>

<h2>Appendix B — Market delta (not an input)</h2>
<p class="lede">Origin vs current market. Highlighted rows differ by ≥1.5 on spread or ≥2.0 on total.</p>
<div class="tblwrap"><table>
<thead><tr><th>Game</th><th>Origin S</th><th>Mkt S</th><th>ΔS</th><th>Origin T</th><th>Mkt T</th><th>ΔT</th></tr></thead>
<tbody>{''.join(mktd)}</tbody>
</table></div>

<h2>Appendix C — Data issues</h2>
<ul class="issues">{''.join(issues)}</ul>

<footer>
<p>Engines: <span class="num">nfelo</span> (nfeloapp.com Week 1 model spreads supplied 2026-09-02; modifiers from the greerreNFL/nfelo 2026-09-01 update; nfelo publishes no total — implied totals derived per §3.2) ·
<span class="num">PFF</span> (Power Rankings point-spread ratings, pff.com/betting/nfl-power-rankings, table supplied 2026-09-01 — authoritative per §12; spread_pff = −(PSR<sub>home</sub> − PSR<sub>away</sub>) − site HFA) ·
<span class="num">TPT</span> panel: Donchess/DRatings + FF-Winners from the user-supplied Week 1 files (Pi-Rate, Lou St. John, RP Excel, Laffaye, Dokter blank); Donchess cross-checked against DRatings' current projections.
League total prior <span class="num">46.0</span> (2025 realized mean).
Full audit trail: <span class="num">betting/experiments/originator/2026-W01/</span> on branch claude/new-session-xoaxrh.</p>
</footer>
</div>
"""

out = RUN / "artifact_origin_card.html"
out.write_text(page)
print(f"wrote {out} ({len(page)/1024:.0f} KB)")
