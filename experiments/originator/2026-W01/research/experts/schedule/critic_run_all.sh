#!/bin/bash
# Re-runs every critic script in order; output to _critic_run_all.log. Expert scripts: run_all.sh (reproduced byte-identical on 2026-09-04).
cd "$(dirname "$0")"
for f in critic_00_hfa_mod.py critic_01_rest.py critic_02_travel.py critic_02b_travel_teams.py critic_03_weeks.py critic_04_asbuilt_oos.py; do
  echo "===== $f ====="; python "$f" 2>&1 | grep -v Warning | grep -v "r = sm.OLS"
done
