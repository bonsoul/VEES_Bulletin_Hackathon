"""
Active AMR surveillance (NVLS) cleaning. One row per isolate x antibiotic.

  - county / sub-county / ward = Kenya shapefile names (clean_admin_names.py)
  - dates are day-first in two styles ('27/01/2025' and '1/4/25'); both parsed
  - species as in ND1 (Bovine -> Cattle, Avian-Domestic -> Chicken); breed kept
  - antibiotic split into name and disc potency ('Ampicillin 10 µg' -> 'Ampicillin', '10 µg')
  - text 'null', '.' and empty cells -> missing; zone diameter (result_mm) numeric,
    never filled with 0; method 'Disc diffusion' -> 'Disk diffusion'

Run from the project root:
    python 02cleaning_scripts/clean_amr.py
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import clean_admin_names
from harmonise import read_raw, standardise_species

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_amr")

ROOT = Path(__file__).resolve().parents[1]

RENAME = {
    "ACCEPTING_LAB": "accepting_lab", "TESTING_LAB": "testing_lab", "TESTING_SECTION": "testing_section",
    "SAMPLING_PURPOSE": "sampling_purpose", "SUBMISSION_NUMBER": "submission_number",
    "SAMPLE_IDENTIFICATION": "sample_id", "DATE_RECEIVED": "date_received", "SAMPLING_DATE": "sampling_date",
    "sub_county": "subcounty", "ORIGINAL_SPECIES": "species_raw", "BREED": "breed", "SAMPLE_TYPE": "sample_type",
    "ORGANISM": "organism", "TEST": "test", "METHOD": "method", "ANTIBIOTIC": "antibiotic_raw",
    "RESULT_DATE": "result_date", "RESULT_1": "result_mm", "amr_status": "amr_status",
    "INTERPRETATION": "interpretation_code", "DATE_REPORTED": "date_reported",
}
OUTPUT_COLS = [
    "sample_id", "submission_number", "accepting_lab", "testing_lab", "sampling_purpose",
    "sampling_date", "date_received", "result_date", "date_reported",
    "county", "county_iso", "subcounty", "ward", "subcounty_raw", "ward_raw", "admin_match",
    "species", "species_raw", "breed", "sample_type", "organism", "test", "method",
    "antibiotic", "disc_potency", "antibiotic_raw", "result_mm", "amr_status", "interpretation_code",
]
ANTIBIOTIC_FIXES = {"Chloramphenical": "Chloramphenicol", "Gentamycin": "Gentamicin"}


def day_first_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return out.fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def split_antibiotic(s: pd.Series) -> pd.DataFrame:
    """'Vancomycin (VA) 30ug' -> ('Vancomycin', '30 µg'); 'Trimethoprim-sulfamethoxazole 1.27/23.75 ug'."""
    txt = s.str.replace(r"\((VA|LZD)\)", "", regex=True).str.replace(r"\(\s*except[^)]*\)", "", regex=True)
    m = txt.str.extract(r"^\s*(.*?)\s*([\d.]+(?:/[\d.]+)?)\s*(µg|ug|mcg)?\s*$")
    name = m[0].fillna(txt).str.strip().str.replace(r"\s+", " ", regex=True).replace(ANTIBIOTIC_FIXES)
    potency = (m[1] + " µg").where(m[1].notna())
    return pd.DataFrame({"antibiotic": name, "disc_potency": potency}, index=s.index)


def run(src: Path, out: Path) -> pd.DataFrame:
    df = read_raw(src)
    df.columns = df.columns.str.strip()
    log.info("Loaded %d rows from %s", len(df), src)
    df = df.rename(columns=RENAME)
    df = df.replace({"null": np.nan, "NULL": np.nan, ".": np.nan})

    for col in ["sampling_date", "date_received", "result_date", "date_reported"]:
        parsed = day_first_date(df[col])
        bad = df[col].notna() & parsed.isna()
        if bad.any():
            log.warning("%s: %d unparseable dates: %s", col, bad.sum(), df.loc[bad, col].unique()[:5].tolist())
        df[col] = parsed

    df["species"] = standardise_species(df["species_raw"].str.replace("-", " "))
    df["breed"] = df["breed"].str.strip().str.capitalize()
    df["method"] = df["method"].str.replace("Disc diffusion", "Disk diffusion", case=False)
    df = df.join(split_antibiotic(df["antibiotic_raw"]))
    df["result_mm"] = pd.to_numeric(df["result_mm"], errors="coerce")
    df["amr_status"] = df["amr_status"].str.strip().str.capitalize()
    for col in ["county", "subcounty", "ward"]:
        df[col] = df[col].str.title()

    df = clean_admin_names(df, lat_col=None, lon_col=None)
    df = df[OUTPUT_COLS].sort_values(["sampling_date", "sample_id", "antibiotic"], kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    log.info("Antibiotics: %s", sorted(df["antibiotic"].dropna().unique()))
    log.info("AMR status: %s", df["amr_status"].value_counts(dropna=False).to_dict())
    log.info("Wrote %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "data/active_amr_surveillance_data_nvls.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/amr_clean.csv")
    args = parser.parse_args()
    run(args.src, args.out)
