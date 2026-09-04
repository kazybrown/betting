#!/bin/bash
# Re-run every HFA expert script in order; output to run_all.log
cd "$(dirname "$0")"
for f in 00_explore.py 00b_nfelo_composition.py 01_league_hfa_by_season.py 01b_hfa_constant_bootstrap.py 02_site_hfa.py 02b_nfelo_tz_and_denver.py 03_primetime_hfa.py 04_neutral_international.py 05_divisional.py 05b_div_totals.py 05c_div_totals_by_season.py; do
  echo "=============== $f"; python3 "$f"
done
