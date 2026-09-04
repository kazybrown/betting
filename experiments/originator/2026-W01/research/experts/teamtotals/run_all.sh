#!/bin/bash
# Re-run every team-total expert script in order. 04a rebuilds the pbp feature file (~3-4 min).
cd "$(dirname "$0")"
for f in 01_identity_regression.py 02_median_vs_mean.py 03_home_away_asymmetry.py 04a_build_pbp_features.py 05a_matchup_pfpa.py 05b_matchup_pbp.py 06_rolling_origin.py 07_noise_cost.py 08_fav_bin_rule.py 09_modern_era_check.py; do
  echo "=================== $f"; python3 "$f" 2>&1 | grep -v -E "Warning|warnings.warn|^\s+\"?qreg|te\[f" ; done
