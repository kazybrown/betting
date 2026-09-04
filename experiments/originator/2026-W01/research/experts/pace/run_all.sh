#!/bin/bash
# Re-runs the whole pace-expert pipeline in order. ~5 minutes. Logs land next to each script.
set -e
cd "$(dirname "$0")"
for s in 00_build_teamgame 01_features_validate 02_pace_totals 03_explosive_teamtotals 04_proe 05_recommendation 06_dispersion; do
  echo "=== $s"; python3 $s.py 2>&1 | grep -v -e PerformanceWarning -e '^  m\["dome"\]' -e 'frame\["[ha]c"\]' -e SettingWithCopy -e '^  tgf\["' > $s.log
done
echo "done"
