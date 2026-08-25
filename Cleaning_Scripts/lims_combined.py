"""
Combine the quarterly VEES Diseases_LIMS .xls files into a single dataset.

Requirements:
    pip install pandas xlrd openpyxl

Usage:
    python combine_vees_lims.py

Edit FOLDER below if your files live somewhere else.
"""

import os
import pandas as pd

# ---- CONFIG: update if your folder path differs ----
FOLDER = r"C:\Users\ADMIN\Documents\VEES_Bulletin_Hackathon\Raw_Datasets"

FILES = [
    "Diseases_LIMS_ January to March 2025.xls",
    "Diseases_LIMS_April to June 2025.xls",
    "Diseases_LIMS_July to December  2025.xls",
    "Diseases_LIMS_January to 18th August 2026.xls",
]

OUTPUT_PATH = os.path.join(FOLDER, "..", "Cleaned Data", "Diseases_LIMS_Combined.csv")
# ------------------------------------------------------


def read_one(folder: str, filename: str) -> pd.DataFrame:
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Check the filename spelling/spacing against what's in the folder."
        )
    df = pd.read_excel(path, engine="xlrd")  # .xls format requires xlrd
    # normalize column names so the same field doesn't split into two
    # columns across quarters just because of case/whitespace differences
    df.columns = [str(c).strip() for c in df.columns]
    df["source_file"] = filename
    print(f"  {filename}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def main():
    print("Reading files...")
    frames = [read_one(FOLDER, f) for f in FILES]

    # Union of all columns is kept automatically; anything missing in a
    # given quarter's file is filled with NaN rather than raising an error.
    disease_lims_combined = pd.concat(frames, ignore_index=True, sort=False)

    # Flag any columns that don't appear in every file, so you can sanity
    # check whether that's a real schema difference or a naming mismatch.
    col_sets = [set(f.columns) for f in frames]
    all_cols = set.union(*col_sets)
    inconsistent = [c for c in all_cols if not all(c in s for s in col_sets)]
    if inconsistent:
        print("\nColumns not present in every file (check for naming drift):")
        for c in sorted(inconsistent):
            print(f"  - {c}")

    print(f"\nCombined dataset: {disease_lims_combined.shape[0]} rows, {disease_lims_combined.shape[1]} columns")

    disease_lims_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {os.path.abspath(OUTPUT_PATH)}")

    return disease_lims_combined


if __name__ == "__main__":
    disease_lims_combined = main()