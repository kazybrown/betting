#!/bin/bash
# Re-runs the critic's analysis of the pace expert in order. critic_00 re-runs the expert's own
# pipeline (00-06, ~5 min) and diffs the logs; critic_01..06 are the attacks (~3 min, placebo loops).
set -e
cd "$(dirname "$0")"
./critic_00_repro.sh > critic_00_repro.log 2>&1
for s in critic_01_T1_pace critic_02_T2_explosive critic_03_T3_proe critic_04_T4_epa critic_05_extra critic_06_T4_rule_variants; do
  echo "=== $s"; python3 $s.py 2>&1 | grep -v -e Warning -e SettingWithCopy -e '^  m\[' -e '^  tp\[' -e '^  frame' > $s.log
done
echo done
