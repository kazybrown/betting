#!/bin/bash
# Re-run every script of the uncertainty expert in order; logs land next to the scripts.
cd /home/user/originator-2026-w01/research || exit 1
for s in 01_setup_sanity 02_theory1_disagreement_vs_abserr 03_theory2_thresholds 04_theory3_who_is_right \
         05_theory4_interval_rule 06_engine_sd_reconstructed 07_sqrt_rule_and_tags 08_engine_sd_pure_2engine \
         09_blend_grid_and_params 10_robustness_open_line_and_seasons 11_open_line_shrink_factor; do
  python3 experts/uncertainty/$s.py > experts/uncertainty/$s.log 2>&1 && echo "ok  $s" || echo "FAIL $s"
done
