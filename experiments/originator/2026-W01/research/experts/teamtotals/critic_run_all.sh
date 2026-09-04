#!/bin/bash
# Re-run every critic script (adversarial re-analysis of the teamtotals expert). Requires _pbp_teamgame.csv
# (built by 04a_build_pbp_features.py; the critic verified the rebuild is byte-identical).
cd "$(dirname "$0")"
for f in critic_01_identity.py critic_02_median.py critic_02b_lowtt_shade.py critic_03_home_away.py critic_05a_pfpa_rolling.py critic_05b_pbp_split.py critic_07_noise.py critic_08_favbin.py critic_08b_favbin_rules.py; do
  echo "=================== $f"; python3 "$f" 2>&1 | grep -v -E "Warning|warnings.warn"; done
