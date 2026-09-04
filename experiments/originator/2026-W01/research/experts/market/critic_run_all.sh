#!/bin/bash
# Re-run every critic script for the 'market' expert; logs saved next to the scripts.
cd /home/user/originator-2026-w01/research || exit 1
for s in critic_01_segments critic_02_steam critic_03_rounding critic_04_halfpoint critic_05_totals_keys critic_06_additional; do
  echo "### $s"; python3 experts/market/$s.py | tee experts/market/$s.log
done
