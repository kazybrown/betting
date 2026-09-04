# ORIGINATOR research kit — data, conventions, and the model under test

Purpose: empirically test theories to improve the ORIGINATOR NFL model (spreads, game
totals, team totals). Every claim must come from code run against the files below.
Save every script you run under research/experts/<your_key>/ so a critic can re-run it.

## Discipline (mandatory)
- Benchmark = the market CLOSING line (spread_line / total_line). A theory "improves the
  model" only if it reduces out-of-sample error vs results, or explains error the market
  does not, with effect sizes and n reported. Report MAE / RMSE, ATS rate vs close where
  relevant, and a confidence interval or p-value.
- Out-of-sample: fit on seasons <= 2021, test on 2022-2025 (or rolling-origin). Never
  report an in-sample fit as a finding.
- Sign conventions (get these right or everything is wrong):
  * nflverse games.csv `spread_line` = expected HOME margin; POSITIVE = home favored.
    `result` = home_score - away_score. `total` = home_score + away_score.
  * ORIGINATOR / nfelo convention: NEGATIVE = home favored. kit.py exposes
    `mkt_spread` already in the ORIGINATOR convention (= -spread_line) and `margin`
    (= home - away). A perfect spread prediction has `pred == -margin`, i.e. error =
    margin + pred.
  * nfelo_games.csv: `nfelo_home_line_close/open` and `home_line_close/open` (market)
    are NEGATIVE = home favored. Team ids there use LAR (LA), OAK (LV, pre-2020 and in
    ids), STL, SD; kit.py normalizes ids on both sides before joining.
- Weather columns exist only for outdoor games (temp F, wind mph); roof in
  {outdoors, dome, closed, open}. Rest = days since last game (7 = normal).

## Files (research/data/)
- games_1999_2025.csv — nflverse schedule/results, 7,276 scored games, closing
  spread/total/moneylines every season, temp/wind (outdoor), roof, surface, rest, QB
  names, coaches, referee, stadium, div_game, location (Home/Neutral).
- nfelo_games.csv — 2009-2025 (+2026 W1), per game: starting_nfelo_home/away (Elo,
  1505 = average; 25 Elo = 1 point), hfa_mod, bye mods, div_game_mod, dif_surface_mod,
  home_time_advantage_mod, home/away_538_qb_adj, nfelo_dif_base, nfelo lines open/close
  (with and without market regression: see historic_projected_spreads.csv), market
  lines open/close, total lines open/close, implied probabilities.
- historic_projected_spreads.csv — recent seasons: home_line_pre_regression vs
  regressed line, market_regression_factor (how nfelo shrinks toward the market).
- nfelo_scored_individual_games.csv — per game Brier / ATS scoring of 538, qbelo,
  nfelo (regressed + unregressed) and the market.
- cole_team_games_2012_2025.csv — Kevin Cole team-game efficiency (score,
  adj_score, drives, plays, success_rate, EPA, pass_rate, pass_over_exp ...). NOTE:
  the Drive export TRUNCATED each season to ~55 games (2012, 2022-2025 only). Use
  for validation only, not as a primary series.
- cole_qb_table_*.csv — Cole adjusted QB efficiency tables (2025 and earlier).
- cole_power_rankings.csv, pff_power_ratings.csv — current (2026 W1) ratings; no history.
- Play-by-play: /home/user/nflscrapR-data/play_by_play_data/regular_season/reg_pbp_2009..2019.csv
  (nflscrapR schema: posteam, defteam, epa, wp, play_type, yards_gained, game_seconds_remaining...)
  and /home/user/NFLkz/data/cache/pbp/play_by_play_2023..2025.parquet (nflfastR schema).
  2020-2022 pbp is NOT available locally (nflverse releases are egress-blocked).

## The model under test (ORIGINATOR, current v5 build)
- Spread = 0.46*nfelo + 0.39*PFF + 0.15*KevinCole (home perspective). PFF and Cole
  spreads are built as -(rating_home - rating_away) - HFA with nfelo's per-game site
  HFA (hfa_mod + home_time_advantage_mod)/25; spec defaults if unavailable: 1.4 dome,
  1.6-2.0 outdoor, 0.5-1.0 international, 0 neutral.
- Rating -> points: 25 Elo per point; PFF/Cole ratings are already points vs average.
- Structural clamp: if |PFF - nfelo| > 4.5 the PFF spread is pulled to nfelo +/- 4.5.
- Totals: NO engine publishes a total. Each engine's implied total =
  league_prior (46.0 = 2025 realized mean) + 0.35*(rating_home + rating_away), blended
  PFF .38 / nfelo .32 / Cole .30. Pace adj +/-1..2 and weather adj (wind 15-20: -1.0,
  21-30: -2..-3, 30+: -3.5..-5; rain/snow -1..-2.5; cold <20F -0.5..-1.5) exist in
  the spec but are unvalidated.
- Team totals: identity home = T/2 - S/2, away = T/2 + S/2, then +/-0.5..1.5
  matchup reallocations, then rounding to .0/.5 with sum preserved.
- Context adjustments (section 5): QB starter->backup -2.0..-4.5; WR1 out -0.5..-1.5 on
  team total; OL/pass-rush mismatch -1..-2 TT and -0.5..-1.5 spread; short week
  -0.6..-1.2; bye +0.5..+1.0; west-to-east 10am kick -0.4..-0.8; caps: sum <= 2.5
  spread, 3.0 total.
- Rounding: half-up to .0/.5 (x.25-x.74 -> .5). No key-number logic.
- Confidence tags: spread SD <=1.2 HIGH, 1.2-2.2 MED, >2.2 LOW; totals 1.8 / 3.0.

## Helper
research/kit.py: `from kit import load_games, load_nfelo, merged, mae, ats` — see docstrings.
Run from /home/user/originator-2026-w01/research (or add it to sys.path).
