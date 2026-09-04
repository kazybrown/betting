#!/bin/bash
# Re-run every script of the 'market' expert; logs saved next to the scripts.
cd /home/user/originator-2026-w01/research || exit 1
for s in 00_sanity 01_segment_efficiency 02_open_to_close 02b_totals_steam_stats 03_key_numbers 03b_halfpoint_cents 04_totals_key_numbers 04b_totals_rounding_ev 05_rounding_policy_ev 06_rounding_backtest_oos; do
  echo "### $s"; python3 experts/market/$s.py | tee experts/market/$s.log
done
