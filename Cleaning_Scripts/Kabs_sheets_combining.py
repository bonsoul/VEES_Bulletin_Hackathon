"""
Combine all ND1 Form sheets (v15-v27) from reports_2026-08-18.xls into one dataset.

Each sheet is a different form version with slightly different columns
(fields were added/renamed over time). This script:
  1. Reads every sheet
  2. Drops blank "Unnamed" columns (leftover spacer columns from the form export)
  3. Tags each row with its source sheet/version
  4. Concatenates everything, aligning by column name (missing fields -> blank)
  5. Saves the result as one combined Excel file
"""

import pandas as pd

SOURCE_FILE = "reports_2026-08-18.xls"
OUTPUT_FILE = "combined_reports.xlsx"

xl = pd.ExcelFile(SOURCE_FILE)

frames = []
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.insert(0, "Source Form Version", sheet)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True, sort=False)

combined.to_excel(OUTPUT_FILE, index=False)

print(f"Combined {len(frames)} sheets -> {len(combined)} rows, {len(combined.columns)} columns")
print(combined["Source Form Version"].value_counts())