export const meta = {
  name: 'originator-research-panel',
  description: 'Expert panel backtests theories to improve the NFL spread / total / team-total model, with adversarial methodology critics and a synthesis',
  phases: [
    { title: 'Test', detail: 'ten domain experts each backtest a theory family against local data' },
    { title: 'Critique', detail: 'one adversarial methodology critic per expert re-runs and tries to break each finding' },
    { title: 'Synthesize', detail: 'prioritized implementation plan from upheld findings' },
  ],
}

const KIT = `RESEARCH KIT: read /home/user/originator-2026-w01/research/DATA_README.md FIRST (data files, sign conventions, the model under test, the discipline rules). Helper: /home/user/originator-2026-w01/research/kit.py (run from that directory or sys.path.insert it): load_games(), load_nfelo(), merged(), mae(), ats(). pandas/numpy/scipy/statsmodels/pyarrow are installed. Save EVERY script you run under /home/user/originator-2026-w01/research/experts/<your_key>/ (numbered, re-runnable, printing the numbers you report) — a critic will re-run them.
RULES: (1) Every number you report must come from code you ran. (2) Benchmark is the market CLOSING line; a change "improves the model" only if it lowers out-of-sample error vs results or explains residual error the market does not. (3) Fit on seasons <= 2021, test on 2022-2025 (or rolling-origin); label anything in-sample as such. (4) Report n, effect size, CI/p-value, and the exact model-parameter change you recommend (or "keep as is"). (5) Sign conventions are in the README — verify with a sanity check before trusting any result (e.g. corr(mkt_spread, margin) must be strongly negative). (6) Prefer simple, robust specifications over clever ones; a theory with n<150 games is INCONCLUSIVE unless the effect is enormous. (7) Do not fabricate PFF or Kevin Cole history — none exists locally; use nfelo Elo as the rating series and say so.`

const FINDINGS = {
  type: 'object',
  properties: {
    expert: { type: 'string' },
    theories: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, hypothesis: { type: 'string' }, method: { type: 'string' },
      data_used: { type: 'string' }, n: { type: 'number' }, metric: { type: 'string' },
      result: { type: 'string' }, effect_size: { type: 'string' }, uncertainty: { type: 'string' },
      out_of_sample: { type: 'boolean' },
      verdict: { type: 'string', enum: ['SUPPORTED', 'REJECTED', 'INCONCLUSIVE'] },
      recommendation: { type: 'string' }, parameter_change: { type: 'string' },
      expected_impact: { type: 'string' }, confidence: { type: 'string', enum: ['HIGH', 'MED', 'LOW'] },
      caveats: { type: 'string' }, script_path: { type: 'string' } },
      required: ['id', 'hypothesis', 'method', 'n', 'metric', 'result', 'out_of_sample', 'verdict', 'recommendation', 'parameter_change', 'confidence', 'script_path'] } },
    summary: { type: 'string' } },
  required: ['expert', 'theories', 'summary'],
}

const CRITIQUE = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: {
      theory_id: { type: 'string' },
      verdict: { type: 'string', enum: ['UPHELD', 'DOWNGRADED', 'OVERTURNED'] },
      reason: { type: 'string' }, rerun_result: { type: 'string' },
      alternative_spec_result: { type: 'string' }, revised_recommendation: { type: 'string' } },
      required: ['theory_id', 'verdict', 'reason', 'rerun_result'] } },
    additional_findings: { type: 'string' }, notes: { type: 'string' } },
  required: ['verdicts', 'notes'],
}

const SYNTHESIS = {
  type: 'object',
  properties: {
    changes: { type: 'array', items: { type: 'object', properties: {
      priority: { type: 'number' }, area: { type: 'string' }, change: { type: 'string' },
      parameter_spec: { type: 'string' }, evidence: { type: 'string' },
      expected_impact: { type: 'string' }, risk: { type: 'string' }, source_theories: { type: 'string' } },
      required: ['priority', 'area', 'change', 'parameter_spec', 'evidence', 'expected_impact', 'source_theories'] } },
    keep_as_is: { type: 'array', items: { type: 'string' } },
    rejected_theories: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
    executive_summary: { type: 'string' } },
  required: ['changes', 'keep_as_is', 'rejected_theories', 'open_questions', 'executive_summary'],
}

const EXPERTS = [
  { key: 'calibration', title: 'Rating-to-points calibration & blend expert', brief: `Theories: (1) Is 25 Elo per point the right nfelo->margin scale? Regress margin on elo_dif_pts (+HFA) by era; test non-linearity for big favorites (|spread|>7, >10) — do ratings over/under-project blowouts? (2) How much does an origin number that ignores the market lose vs the closing line: MAE of nfelo unregressed vs regressed (historic_projected_spreads.csv) vs market; what shrinkage weight w in w*nfelo+(1-w)*market minimizes OOS MAE (informational: the spec forbids market inputs, but the number tells us how far to trust engines). (3) Is the structural clamp (pull PFF within 4.5 of nfelo) sensible — using nfelo-vs-market disagreement as a proxy, do large engine disagreements predict which side is right? (4) Does nfelo's own QB adjustment (home_net_qb_mod) add OOS accuracy beyond base Elo?` },
  { key: 'hfa', title: 'Home-field advantage expert', brief: `Theories: (1) League HFA by season 2009-2025 (margin minus rating-implied margin, and market spread residual): has it declined, and what value should the PFF/Cole paths use in 2026? (2) Site-specific HFA: are SEA, DEN, KC, GB, NO, BAL etc. reliably above/below league HFA after shrinkage (empirical Bayes) — and does nfelo's hfa_mod capture it (compare nfelo hfa_pts to realized residuals by stadium)? (3) Primetime / Thursday / Monday / Wednesday-Saturday and Week-1 HFA vs Sunday-afternoon. (4) Neutral-site and international games (location == Neutral; London/Germany/Mexico/Brazil/Australia via stadium): realized "home" edge for the designated home team. (5) Divisional games: smaller HFA / smaller margins?` },
  { key: 'totals', title: 'Game-totals modeler (incl. environment)', brief: `The model's biggest weakness: no engine publishes a total, so it uses implied_total = 46.0 + 0.35*(home_pts_vs_avg + away_pts_vs_avg). Theories: (1) Fit total_pts ~ a + b*(home_pts_vs_avg + away_pts_vs_avg) using nfelo Elo; is b ~0.35? Is combined strength even predictive of totals, or should offense/defense be separated (use nflverse scoring history: rolling points-for/points-against per team as offense/defense proxies)? (2) Build the best simple OOS totals model you can from local data (rolling team PF/PA, prior-season totals, pace from pbp where available, roof, div_game, rest, week-of-season, temp/wind) and report its OOS MAE vs the market total's MAE — how close can an originator get, and which features matter? (3) Weather: quantify wind (thresholds 10/15/20/25 mph), temp (<32F, <20F), and precipitation proxies on totals and on spreads (favorites in wind?), validating or replacing the spec's weather table. (4) Dome/closed-roof effect on totals after controlling for teams. (5) Recommend the totals formula and weather table.` },
  { key: 'teamtotals', title: 'Team-total allocation expert', brief: `The model splits team totals by the identity home = T/2 - S/2, away = T/2 + S/2, then applies +/-0.5..1.5 matchup reallocations. Theories: (1) Regress home_score and away_score on (mkt_spread, mkt_total): are the coefficients 0.5/-0.5 and 0.5/+0.5, or does the favorite's share of points differ from linear (e.g. big favorites score more than T/2 - S/2 implies)? Fit separately for favorites/dogs and by spread size, OOS. (2) Home/away asymmetry at fixed spread (do home teams score more of the total than the identity implies?). (3) Is there evidence for the spec's matchup reallocations — explosive offense vs weak defense, red-zone defense, pace mismatch — using rolling PF/PA and pbp explosive-play rates (2009-2019, 2023-2025) as proxies? Quantify the magnitude a reallocation should have. (4) Recommend the allocation formula and the reallocation ranges.` },
  { key: 'qb', title: 'Quarterback & injury-pricing expert', brief: `Theories: (1) Value of a backup QB start: using games.csv home/away_qb_name, define each team-season's primary starter (most starts); for games started by a non-primary QB, measure (a) market closing-line movement vs the team's typical rating-implied line and (b) realized margin residual vs the nfelo line WITHOUT QB adjustment (nfelo_dif_base) — what is the points value of starter->backup, and does the spec's -2.0..-4.5 hold? Split by era (2009-2015 / 2016-2025). (2) Does nfelo's 538-style QB adjustment fully price backups (residual after home_net_qb_mod)? (3) Week 1 specifically: new-starter and rookie-starter effects on margin residual vs market. (4) Injury proxies beyond QB are not in local data — say so; recommend the QB parameter values and the double-count rule.` },
  { key: 'schedule', title: 'Rest, travel & schedule-spot expert', brief: `Theories, all as residuals vs the market closing spread AND vs the nfelo line (to separate 'market already prices it' from 'engines miss it'): (1) Short rest: Thursday games after a Sunday game (rest 4 days) — spec says -0.6..-1.2 for the short-rest team; measure (both teams are usually short). (2) Bye advantage (rest >= 13) vs opponent on normal rest; the spec's +0.5..+1.0. (3) Rest differential generally (home_rest - away_rest) as a continuous effect. (4) Travel: west-coast teams playing 1pm ET kicks in the east (use gametime, team home time zones from a small table you define), east-coast teams playing late-window games in the west; cross-country trips. (5) Week-of-season effects: Week 1 (rating uncertainty), Week 18 (rest/motivation). (6) Recommend which schedule adjustments the engines already price (so the spec should NOT add them) and the sizes for the rest.` },
  { key: 'market', title: 'Market microstructure & key-numbers expert', brief: `Theories: (1) Closing-line efficiency by segment 2009-2025: spread error by favorite size, home/away favorite, primetime, divisional, Week 1, playoffs; total error by total size and roof. Where does the market miss systematically (if anywhere) — those are the places an originator can legitimately differ. (2) Open-to-close movement (home_line_open vs home_line_close in nfelo_games.csv): does the direction of the move predict results beyond the close (i.e. should an originator lean with steam)? (3) Key numbers: distribution of final margins at 3, 7, 6, 10, 4, 1; push probabilities; given that, what rounding policy for an origin spread landing at x.25/x.75 near 3 or 7 maximizes expected value vs the market — compare the spec's half-up rule with a key-number-aware rule. (4) Half-point value table around 3 and 7 (empirical). (5) Totals key numbers (41, 44, 47, 51) — do they matter enough to change rounding? Recommend the rounding rule.` },
  { key: 'uncertainty', title: 'Uncertainty & confidence-tag expert', brief: `The model tags confidence from engine disagreement (spread SD <=1.2 HIGH, 1.2-2.2 MED, >2.2 LOW; totals 1.8/3.0) but has never validated that disagreement predicts error. Theories: (1) Using |nfelo close - market close| (and |nfelo open - nfelo close|) as the disagreement proxy, does larger disagreement predict larger absolute error of the nfelo number AND of the market number? Fit |error| ~ disagreement OOS; report the slope and calibration by tercile. (2) Are the SD thresholds sensible — what disagreement levels correspond to a 10th/50th/90th-percentile error? (3) Does disagreement predict WHO is right (market or model) — i.e. should the blend weight on the model shrink when engines disagree? (4) Build a predictive-interval rule: recommend a formula for a per-game error SD (spread and total) from available inputs (disagreement, favorite size, total size, roof, week) and the tag thresholds that make HIGH/MED/LOW mean something.` },
  { key: 'pace', title: 'Pace, play-style & explosiveness expert', brief: `Using play-by-play (nflscrapR 2009-2019 CSVs and nflfastR 2023-2025 parquet; note 2020-2022 missing) plus Kevin Cole's truncated team-game table for validation: Theories: (1) Compute per-team-season and rolling (prior 8 games) pace (plays per game, seconds per play in neutral situations), neutral-script pass rate, and explosive-play rate (>=20-yd pass / >=10-yd rush) and points per drive. Does prior pace of both teams predict the realized total beyond the market total (residual regression, OOS)? Quantify: what total adjustment does a fast/fast vs slow/slow matchup warrant — the spec says +/-1..2; validate or replace. (2) Does explosive-offense vs weak-explosive-defense predict team scoring beyond the market implied team total (T/2 -/+ S/2)? (3) Pass-rate-over-expected as a totals signal. (4) Recommend the pace/style adjustment sizes and the exact features to compute each week from pbp.` },
  { key: 'earlyseason', title: 'Early-season & preseason-rating expert', brief: `Week 1 is the current card. Theories: (1) How predictive are preseason ratings in Weeks 1-4 vs Weeks 5+? Using nfelo's starting Elo (regressed preseason values) and the market: error of nfelo and of the market by week-of-season; is the Week 1 market less efficient (larger errors, exploitable biases: favorites, home teams, totals over/under)? (2) Should an originator shrink its Week 1 numbers toward zero (spread) or the league mean (total) — fit the optimal shrinkage factor by week OOS. (3) Week 1 totals: realized vs market by season — systematic under/over? (4) New head coach / new starting QB (from games.csv coach and qb name changes vs prior season) in Week 1: residual vs market. (5) Recommend Week-1-specific parameters (shrinkage, HFA, total prior, adjustment caps).` },
]

phase('Test')

const results = await pipeline(
  EXPERTS,
  (e) => agent(`${KIT}

ROLE: ${e.title} on a panel improving the ORIGINATOR NFL spread/total/team-total model. Your key is "${e.key}" (scripts go in research/experts/${e.key}/).
${e.brief}
Work through every theory listed with real code and real backtests. For each, fill the schema honestly — REJECTED and INCONCLUSIVE are valuable outcomes. Recommendations must be concrete parameter changes the engine can implement (a formula, a table, a threshold) or an explicit "keep as is". Finish with a summary of what you would change first and why.`,
    { label: `expert:${e.key}`, phase: 'Test', schema: FINDINGS }),

  (findings, e) => findings && agent(`${KIT}

ROLE: ADVERSARIAL METHODOLOGY CRITIC for the "${e.title}" (key ${e.key}). Your default stance is that each finding is wrong until it survives you. For EVERY theory below: re-run the expert's script from research/experts/${e.key}/ (fix the path if needed) and confirm the numbers; then attack it — look for sign-convention errors (see README), leakage (using information not available before kickoff, e.g. closing lines to predict closing lines, current-season aggregates), in-sample fitting reported as OOS, small n, multiple-comparison fishing, survivorship, confounds (team quality, era, roof), and mis-specified benchmarks. Try at least one alternative specification per SUPPORTED theory (different seasons, rolling-origin, robust regression, a placebo). Save your scripts under research/experts/${e.key}/critic_*.py. Verdict per theory: UPHELD (survives, numbers reproduce), DOWNGRADED (real but smaller / less certain — give the revised recommendation), OVERTURNED (wrong or unsupported — say why). Add any additional finding your re-analysis produced.

EXPERT FINDINGS:
${JSON.stringify(findings, null, 1)}`,
    { label: `critic:${e.key}`, phase: 'Critique', schema: CRITIQUE })
      .then(critique => ({ expert: e.key, title: e.title, findings, critique }))
)

const kept = results.filter(Boolean)
log(`Test+Critique complete for ${kept.length}/${EXPERTS.length} experts`)

phase('Synthesize')
const synthesis = await agent(`${KIT}

ROLE: SYNTHESIS LEAD. Below are ten experts' backtested findings and the adversarial critiques of each. Produce the implementation plan for the ORIGINATOR model:
- changes: ONLY from theories UPHELD or DOWNGRADED by the critic, ordered by expected out-of-sample improvement; each with an exact parameter spec the engine can code (formula, table, threshold, weight), the evidence (numbers + n + OOS), expected impact, and risk. Where two experts touch the same mechanism (e.g. HFA in the rating paths vs schedule adjustments; pace in totals vs team totals), reconcile them into ONE change so nothing is double-counted.
- keep_as_is: model elements the evidence says not to change (and why).
- rejected_theories: with the one-line reason.
- open_questions: what could not be tested locally (missing data) and what data would settle it.
- executive_summary: 8-12 sentences for the model owner — numbers first, no hype.

EXPERT + CRITIC RESULTS:
${JSON.stringify(kept.map(k => ({ expert: k.expert, title: k.title, theories: k.findings.theories.map(t => ({ id: t.id, hypothesis: t.hypothesis, n: t.n, result: t.result, effect_size: t.effect_size, uncertainty: t.uncertainty, out_of_sample: t.out_of_sample, verdict: t.verdict, recommendation: t.recommendation, parameter_change: t.parameter_change, expected_impact: t.expected_impact, confidence: t.confidence, caveats: t.caveats })), expert_summary: k.findings.summary, critic: k.critique })), null, 1)}`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTHESIS })

return { experts: kept, synthesis }
