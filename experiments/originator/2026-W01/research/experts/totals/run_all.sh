#!/bin/bash
# Re-run the totals expert's scripts in order. 00 rebuilds the pbp pace cache (~1 min).
set -e
cd "$(dirname "$0")"
for s in 00_build_pace 01_elo_totals 02_totals_model 03_weather 04_dome 05_recommendation 06_sensitivity; do
  echo "=============== $s ==============="
  python3 $s.py 2>&1 | tee $s.log
done
