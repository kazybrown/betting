#!/bin/bash
# Re-run every early-season expert script in order; each writes its own .log and everything goes to _run_all.log
cd "$(dirname "$0")"
for f in 00_sanity.py 01_rating_power_by_week.py 02_shrinkage_oos.py 03_week1_totals.py 04_new_coach_qb.py 05_week1_hfa_caps.py 06_recommendation.py; do
  echo "=============== $f"; python3 "$f" | tee "${f%.py}.log"
done 2>&1 | tee _run_all.log
