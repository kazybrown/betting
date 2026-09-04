#!/bin/bash
# critic_00: re-run the pace expert's whole pipeline (00-06) unchanged, writing critic_repro_XX.log,
# then diff each against the expert's own XX.log (same grep filter as run_all.sh). Also checks that the
# intermediate CSVs are byte-identical after the re-run (determinism of the feature build).
set -e
cd "$(dirname "$0")"
md5sum _teamgame.csv _teamgame_feats.csv _game_features.csv > critic_repro_md5_before.txt
for s in 00_build_teamgame 01_features_validate 02_pace_totals 03_explosive_teamtotals 04_proe 05_recommendation 06_dispersion; do
  echo "=== $s"; python3 $s.py 2>&1 | grep -v -e PerformanceWarning -e '^  m\["dome"\]' -e 'frame\["[ha]c"\]' -e SettingWithCopy -e '^  tgf\["' > critic_repro_$s.log
  if diff -q $s.log critic_repro_$s.log > /dev/null; then echo "  $s.log: IDENTICAL"; else echo "  $s.log: DIFFERS"; diff $s.log critic_repro_$s.log | head -40; fi
done
md5sum _teamgame.csv _teamgame_feats.csv _game_features.csv > critic_repro_md5_after.txt
echo "--- md5 before/after:"; diff critic_repro_md5_before.txt critic_repro_md5_after.txt && echo "CSVs byte-identical"
echo "done"
