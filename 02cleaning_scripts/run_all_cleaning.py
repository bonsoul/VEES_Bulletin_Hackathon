"""
Run every VEES cleaning script in order. All of them share the same rules:
  clean_admin_names.py  county / sub-county / ward -> Kenya shapefile names
  harmonise.py          disease, sign and species names, dates, GPS checks

Run from the project root:
    python 02cleaning_scripts/run_all_cleaning.py
    python 02cleaning_scripts/run_all_cleaning.py --only zero_report_cleaning clean_amr

ND1 needs Cleaned Data/nd1_final.csv from 00cleaning.py (run first when raw ND1 exports change).
The admin lookup only needs rebuilding (build_admin_lookup.py) when the boundary files change.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "00cleaning.py",                  # ND1: merge KABS 1.0 + 2.0 -> nd1_final.csv
    "automated_cleaning_script.py",   # ND1: admin, disease, syndrome names -> nd1_final_clean.csv
    "zero_report_cleaning.py",
    "clean_wildlife.py",
    "clean_meat_inspection.py",
    "clean_vaccination.py",
    "clean_community_reports.py",
    "clean_amr.py",
    "clean_population.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="script names (with or without .py) to run")
    args = parser.parse_args()
    wanted = {s if s.endswith(".py") else f"{s}.py" for s in args.only} if args.only else None
    failed = []
    for script in SCRIPTS:
        if wanted and script not in wanted:
            continue
        start = time.time()
        print(f"\n=== {script}", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE.parent)
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"=== {script}: {status} in {time.time() - start:.1f}s", flush=True)
        if result.returncode:
            failed.append(script)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
