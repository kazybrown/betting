#!/bin/bash
# Re-runs every schedule-expert script in order; output to _run_all.log
cd "$(dirname "$0")"
for f in 00_explore.py 00b_decomposition_check.py common.py 01_short_rest.py 02_bye.py 03_rest_diff.py 04_travel.py 04b_travel_robustness.py 04c_travel_rules_oos.py 05_week_effects.py 06_recommendation.py; do
  echo "===== $f ====="; python "$f" 2>&1 | grep -v Warning | grep -v "r = sm.OLS"
done
