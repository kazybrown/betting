#!/usr/bin/env python3
"""Render the expert-panel research report as an HTML artifact page."""

import html
import json
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parent
raw = json.loads((RUN / "research" / "panel_results_raw.json").read_text())
synth_p = RUN / "research" / "synthesis.json"
synth = json.loads(synth_p.read_text()) if synth_p.exists() else None
cmp_p = RUN / "final_research.json"
cmp = json.loads(cmp_p.read_text()) if cmp_p.exists() else []
cfg = json.loads((RUN / "research_config.json").read_text())
GEN = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True).stdout.strip()

TITLES = {
    "calibration": "Rating-to-points calibration & blend", "hfa": "Home-field advantage",
    "totals": "Game totals & environment", "teamtotals": "Team-total allocation",
    "qb": "Quarterback & injury pricing", "schedule": "Rest, travel & schedule spots",
    "market": "Market microstructure & key numbers", "uncertainty": "Uncertainty & confidence tags",
    "pace": "Pace, play-style & explosiveness", "earlyseason": "Early season & preseason ratings",
}


def esc(s):
    return html.escape(str(s if s is not None else ""))


def chip(v):
    cls = {"UPHELD": "ok", "SUPPORTED": "ok", "DOWNGRADED": "mid", "INCONCLUSIVE": "mid",
           "OVERTURNED": "bad", "REJECTED": "bad"}.get(v, "mid")
    return f'<span class="chip {cls}">{esc(v)}</span>'


# ---- critic verdict lookup
crit = {}
for c in raw["critics"]:
    for v in c["verdicts"]:
        crit[v["theory_id"]] = v

# ---- expert sections
sections = []
for e in raw["experts"]:
    key = next((k for k in TITLES if k in str(e.get("expert", "")).lower()), None) or e.get("expert", "?")
    title = TITLES.get(key, e.get("expert", "?"))
    rows = []
    for t in e["theories"]:
        cv = crit.get(t["id"], {})
        rows.append(f"""<tr>
<td class="hyp"><b>{esc(t['id'])}</b><br>{esc(t['hypothesis'])}</td>
<td class="num">{esc(t.get('n', ''))}</td>
<td class="res">{esc(t['result'])}</td>
<td>{chip(t['verdict'])}<br><span class="mut small">{'OOS' if t.get('out_of_sample') else 'in-sample'} · conf {esc(t.get('confidence',''))}</span></td>
<td>{chip(cv.get('verdict', '—'))}<br><span class="small">{esc(cv.get('reason', ''))[:420]}</span></td>
<td class="rec">{esc(cv.get('revised_recommendation') or t.get('parameter_change', ''))}</td>
</tr>""")
    sections.append(f"""<details class="expert" open>
<summary><span class="g">{esc(title)}</span><span class="mut small">{len(e['theories'])} theories · scripts in research/experts/{esc(key)}/</span></summary>
<div class="body">
<p class="lede-in">{esc(e.get('summary', ''))}</p>
<div class="tblwrap"><table>
<thead><tr><th>Theory</th><th>n</th><th>Result</th><th>Expert</th><th>Critic</th><th>Recommendation (critic-revised)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
</div></details>""")

# ---- synthesis
synth_html = "<p class=\"mut\">Synthesis pending.</p>"
if synth:
    ch = "".join(f"""<li><div class="chg-head"><span class="pri">{esc(c['priority'])}</span><b>{esc(c['area'])}</b> — {esc(c['change'])}</div>
<div class="chg-body"><p><span class="lab">Parameter</span> <code>{esc(c['parameter_spec'])}</code></p>
<p><span class="lab">Evidence</span> {esc(c['evidence'])}</p>
<p><span class="lab">Expected impact</span> {esc(c['expected_impact'])} <span class="mut">· risk: {esc(c.get('risk',''))} · from {esc(c.get('source_theories',''))}</span></p></div></li>"""
                 for c in sorted(synth["changes"], key=lambda x: x["priority"]))
    keep = "".join(f"<li>{esc(k)}</li>" for k in synth["keep_as_is"])
    rej = "".join(f"<li>{esc(k)}</li>" for k in synth["rejected_theories"])
    opn = "".join(f"<li>{esc(k)}</li>" for k in synth["open_questions"])
    synth_html = f"""<p class="exec">{esc(synth['executive_summary'])}</p>
<h3>Recommended changes, by expected out-of-sample improvement</h3>
<ol class="changes">{ch}</ol>
<div class="cols">
<div><h3>Keep as is</h3><ul class="plain">{keep}</ul></div>
<div><h3>Rejected theories</h3><ul class="plain">{rej}</ul></div>
<div><h3>Open questions (data needed)</h3><ul class="plain">{opn}</ul></div>
</div>"""

# ---- side-by-side comparison
cmp_rows = "".join(f"""<tr><td class="g">{g['away']}@{g['home']}</td>
<td class="num">{g['home']} {g['spread_origin_spec']:+.1f}</td><td class="num strong">{g['home']} {g['spread_origin_r']:+.1f}</td><td class="num mut">{g['market_spread']:+.1f}</td>
<td class="num">{g['total_origin_spec']:.1f}</td><td class="num strong">{g['total_origin_r']:.1f}</td><td class="num mut">{g['market_total']:.1f}</td>
<td class="num">{g['tt_home_spec']:.1f}/{g['tt_away_spec']:.1f}</td><td class="num strong">{g['tt_home_r']:.1f}/{g['tt_away_r']:.1f}</td>
<td class="num">{g['gap_spread']:+.1f} / {g['gap_total']:+.1f}</td><td class="num">{g['expected_real_gap_spread']:+.2f} · {g['excess_rmse_spread']:.2f}/{g['excess_rmse_total']:.2f}</td>
<td>{chip(g['conf_spread_r'])}{chip(g['conf_total_r'])}</td></tr>""" for g in cmp)

t = cfg["totals"]
page = f"""<title>Originator Research Panel</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root {{ --ground:#F6F7F4; --surface:#FFFFFF; --ink:#1A2420; --mut:#5C6B64; --accent:#0E6B4F; --amber:#9A6A00;
  --rule:#D9DED9; --rowline:#E7EAE6; --ok:#0E6B4F; --ok-bg:#E3F0EA; --mid:#7A5C00; --mid-bg:#F4ECD4; --bad:#8C3A2B; --bad-bg:#F6E4DF; --code:#EEF1EC; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ground:#101614; --surface:#18201C; --ink:#E4EAE5; --mut:#8CA096; --accent:#4FC08D; --amber:#D9A62E;
  --rule:#2A3630; --rowline:#222C27; --ok:#4FC08D; --ok-bg:#173327; --mid:#D9A62E; --mid-bg:#33290F; --bad:#E08A6D; --bad-bg:#33110A; --code:#1F2925; }} }}
:root[data-theme="dark"] {{ --ground:#101614; --surface:#18201C; --ink:#E4EAE5; --mut:#8CA096; --accent:#4FC08D; --amber:#D9A62E;
  --rule:#2A3630; --rowline:#222C27; --ok:#4FC08D; --ok-bg:#173327; --mid:#D9A62E; --mid-bg:#33290F; --bad:#E08A6D; --bad-bg:#33110A; --code:#1F2925; }}
body {{ background:var(--ground); color:var(--ink); font:15px/1.55 "IBM Plex Sans","Segoe UI",system-ui,sans-serif; margin:0; padding:0 20px 72px; }}
.wrap {{ max-width:1120px; margin:0 auto; }}
header {{ padding:40px 0 20px; border-bottom:2px solid var(--ink); }}
.eyebrow {{ font:600 12px/1 "IBM Plex Mono",monospace; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); margin:0 0 10px; }}
h1 {{ font:600 34px/1.1 "IBM Plex Sans Condensed","Arial Narrow",sans-serif; margin:0 0 8px; text-wrap:balance; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 22px; font:13px/1.4 "IBM Plex Mono",monospace; color:var(--mut); }}
h2 {{ font:600 21px/1.2 "IBM Plex Sans Condensed","Arial Narrow",sans-serif; margin:44px 0 8px; }}
h3 {{ font:600 12px/1 "IBM Plex Mono",monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--mut); margin:22px 0 8px; }}
p.lede, p.lede-in {{ color:var(--mut); font-size:13.5px; max-width:72ch; margin:0 0 14px; }}
p.exec {{ max-width:75ch; font-size:16px; line-height:1.6; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px; background:var(--rule); border:1px solid var(--rule); margin:18px 0 6px; }}
.stat {{ background:var(--surface); padding:12px 14px; }}
.stat .v {{ font:600 24px/1.1 "IBM Plex Mono",monospace; color:var(--accent); }}
.stat .k {{ font:600 11px/1.3 "IBM Plex Mono",monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); margin-top:4px; }}
ol.changes {{ list-style:none; padding:0; margin:0; counter-reset:none; }}
ol.changes li {{ border:1px solid var(--rule); background:var(--surface); margin:0 0 10px; padding:12px 16px; }}
.chg-head {{ display:flex; gap:12px; align-items:baseline; font-size:15.5px; }}
.pri {{ font:600 13px/1 "IBM Plex Mono",monospace; color:var(--accent); border:1px solid var(--accent); padding:3px 7px; border-radius:2px; flex:none; }}
.chg-body {{ padding:6px 0 0 44px; font-size:13.5px; }}
.chg-body p {{ margin:6px 0; }}
.lab {{ font:600 10.5px/1 "IBM Plex Mono",monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--mut); margin-right:6px; }}
code {{ font:13px "IBM Plex Mono",monospace; background:var(--code); padding:1px 5px; border-radius:2px; }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:24px; margin-top:8px; }}
ul.plain {{ margin:0; padding-left:18px; font-size:13.5px; }} ul.plain li {{ margin:5px 0; }}
.tblwrap {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; min-width:900px; }}
th {{ font:600 11px/1.3 "IBM Plex Mono",monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); text-align:left; padding:9px 10px; border-bottom:1px solid var(--ink); white-space:nowrap; }}
td {{ padding:9px 10px; border-bottom:1px solid var(--rowline); vertical-align:top; font-size:13px; }}
td.hyp {{ min-width:220px; }} td.res {{ min-width:260px; }} td.rec {{ min-width:220px; }}
.num {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.strong {{ font-weight:600; color:var(--accent); }} .mut {{ color:var(--mut); }} .small {{ font-size:12px; }} .g {{ font-weight:600; }}
.chip {{ display:inline-block; font:600 10px/1 "IBM Plex Mono",monospace; letter-spacing:.08em; padding:3px 6px; border-radius:2px; margin:0 4px 3px 0; }}
.chip.ok {{ color:var(--ok); background:var(--ok-bg); }} .chip.mid {{ color:var(--mid); background:var(--mid-bg); }} .chip.bad {{ color:var(--bad); background:var(--bad-bg); }}
details.expert {{ border:1px solid var(--rule); background:var(--surface); margin:0 0 10px; }}
details.expert summary {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; padding:11px 14px; cursor:pointer; list-style:none; }}
details.expert summary::-webkit-details-marker {{ display:none; }}
details.expert summary::before {{ content:"–"; font:600 14px/1 "IBM Plex Mono",monospace; color:var(--mut); }}
details.expert:not([open]) summary::before {{ content:"+"; }}
details.expert .body {{ padding:0 14px 14px; }}
summary:focus-visible, a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
footer {{ margin-top:52px; padding-top:16px; border-top:1px solid var(--rule); font:12.5px/1.6 "IBM Plex Mono",monospace; color:var(--mut); }}
</style>
<div class="wrap">
<header>
<p class="eyebrow">Backtested · adversarially reviewed</p>
<h1>ORIGINATOR model research — expert panel findings</h1>
<div class="meta"><span>Generated {GEN}</span><span>Data: nflverse results 1999–2025 · nfelo lines 2009–2025 · play-by-play 2009–19, 2023–25</span><span>Fit ≤ 2021 · test 2022–25</span></div>
<div class="stats">
<div class="stat"><div class="v">{len(raw['experts'])}</div><div class="k">domain experts</div></div>
<div class="stat"><div class="v">{sum(len(e['theories']) for e in raw['experts'])}</div><div class="k">theories backtested</div></div>
<div class="stat"><div class="v">{sum(1 for v in crit.values() if v['verdict']=='UPHELD')}</div><div class="k">upheld by critics</div></div>
<div class="stat"><div class="v">{sum(1 for v in crit.values() if v['verdict']=='DOWNGRADED')}</div><div class="k">downgraded</div></div>
<div class="stat"><div class="v">{sum(1 for v in crit.values() if v['verdict']=='OVERTURNED')}</div><div class="k">overturned</div></div>
<div class="stat"><div class="v">{len(synth['changes']) if synth else '—'}</div><div class="k">changes recommended</div></div>
</div>
</header>

<h2>Synthesis</h2>
<p class="lede">Only theories the adversarial critic upheld or downgraded feed the plan. Every expert wrote re-runnable scripts; every critic re-ran them and tried alternative specifications before ruling.</p>
{synth_html}

<h2>Week 1 side by side — spec build vs research-calibrated build</h2>
<p class="lede">Same audited §5 adjustments in both. The research build applies the synthesis parameters: rating-path HFA {cfg['spreads']['hfa_rating_paths']} (international neutral {cfg['spreads']['hfa_neutral_international']}); totals = Week-1 prior {t['L_prev'] + t['week1_prior_offset']} + nfelo path ({t['nfelo_path']['b_R']}·QB-free rating + {t['nfelo_path']['b_gt']}·team scoring-rate dev + {t['nfelo_path']['b_qb']}·QB) / rating paths ({t['rating_paths']['b_R']}·rating) + divisional ({t['div']['div_game']:+} / {t['div']['non_div']:+}) + environment (dome {t['env']['dome']:+}, outdoor {t['env']['outdoor_base']:+} at mean wind) + capped EPA term, blended {t['blend']['pff_implied']}/{t['blend']['nfelo_implied']}/{t['blend']['cole_implied']}; matchup reallocations zeroed; Week 1 caps {cfg['week1_caps']['spread']}/{cfg['week1_caps']['total']}. Tags measure distance to the latest market line as expected excess RMSE (diagnostic only — the market is never an input to the number). "Real gap" = λ 0.27 × raw gap, the share of an origin-vs-market gap that survives out of sample.</p>
<div class="tblwrap"><table>
<thead><tr><th>Game</th><th>Spread spec</th><th>Spread research</th><th>Mkt</th><th>Total spec</th><th>Total research</th><th>Mkt</th><th>TT spec</th><th>TT research</th><th>Gap S / T</th><th>Real gap · excess RMSE S/T</th><th>Tag S·T</th></tr></thead>
<tbody>{cmp_rows}</tbody></table></div>

<h2>Expert findings, theory by theory</h2>
<p class="lede">Expert verdict on the left, critic verdict on the right. Sample sizes are games unless stated; "OOS" means the number was measured on held-out seasons.</p>
{''.join(sections)}

<footer>
<p>Method: ten experts, each assigned a theory family, wrote and ran backtests against the local kit (research/DATA_README.md); one adversarial critic per expert re-ran every script, hunted for sign errors, leakage, in-sample reporting and small-n effects, and tried alternative specifications; a synthesis lead reconciled overlapping mechanisms. No PFF or Kevin Cole history exists locally — nfelo's Elo is the rating series throughout. Scripts and logs: betting/experiments/originator/2026-W01/research/experts/.</p>
</footer>
</div>
"""
(RUN / "artifact_research_report.html").write_text(page)
print(f"wrote artifact_research_report.html ({len(page)//1024} KB); synthesis {'present' if synth else 'pending'}; comparison rows {len(cmp)}")
