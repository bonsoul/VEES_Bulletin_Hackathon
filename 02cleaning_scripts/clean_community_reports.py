"""
Community (KABS) disease reports cleaning. One row per report x sign, as in the source;
report_id (the form's Id) links the sign rows of one report.

  - county / sub-county / ward = Kenya shapefile names (clean_admin_names.py)
  - dates are month-first ('7/23/26'); GPS checked
  - species as in ND1; signs harmonised with the shared SIGNS dictionary
    (incl. Swahili: Kuhara -> Diarrhoea, Machozi -> Lacrimation, Kutapika -> Vomiting ...).
    'Others' is replaced by the free-text 'Specify Other Sign'; signs_raw keeps the original
  - herd size / sick / dead are numbers; blanks stay blank (never 0)
  - submitter name / username / UUID are dropped (personal data); organisation and role kept

Run from the project root:
    python 02cleaning_scripts/clean_community_reports.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from clean_admin_names import clean_admin_names
from harmonise import harmonise_signs_cell, read_raw, standardise_species, validate_coords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_community_reports")

ROOT = Path(__file__).resolve().parents[1]

RENAME = {
    "Id": "report_id", "Date of Report": "report_date", "Date of Start of sickness": "sickness_start_date",
    "sub_county": "subcounty", "Village": "village", "Animals Affected": "species_raw",
    "Total Number of Animals in the herd": "herd_size", "Number Sick": "number_sick", "Number Dead": "number_dead",
    "Submitter Organization": "submitter_organisation", "Submitter Role": "submitter_role",
}
OUTPUT_COLS = [
    "report_id", "report_date", "sickness_start_date", "county", "county_iso", "subcounty", "ward",
    "subcounty_raw", "ward_raw", "admin_match", "village", "latitude", "longitude", "coords_in_county",
    "species", "sign", "signs_raw", "herd_size", "number_sick", "number_dead",
    "submitter_organisation", "submitter_role",
]


def run(src: Path, out: Path) -> pd.DataFrame:
    df = read_raw(src).rename(columns=RENAME)
    log.info("Loaded %d rows (%d reports) from %s", len(df), df["report_id"].nunique(), src)

    for col in ["report_date", "sickness_start_date"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")
    late = df["sickness_start_date"] > df["report_date"]
    if late.any():
        log.warning("%d rows have sickness start after the report date (left as-is)", late.sum())

    df = validate_coords(df)
    species = df["species_raw"].where(df["species_raw"].str.lower() != "other", df["Other Species"])
    df["species"] = standardise_species(species)

    is_other = df["Signs of Disease"].str.lower().isin(["others", "other"])
    df["signs_raw"] = df["Signs of Disease"].where(~is_other, df["Specify Other Sign"]).fillna(df["Signs of Disease"])
    df["sign"] = df["signs_raw"].map(harmonise_signs_cell)

    for col in ["herd_size", "number_sick", "number_dead"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    bad = (df["number_sick"] > df["herd_size"]) | (df["number_dead"] > df["herd_size"])
    if bad.any():
        log.warning("%d rows have sick or dead > herd size (left as-is)", bad.sum())
    df["village"] = df["village"].str.strip().str.title()

    df = clean_admin_names(df)
    df = df[OUTPUT_COLS].sort_values(["report_date", "report_id"], kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    signs = df["sign"].str.split("; ").explode()
    log.info("Signs (%d distinct): %s", signs.nunique(), signs.value_counts().head(15).to_dict())
    log.info("Wrote %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "data/community_reports_kabs_separated.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/community_reports_clean.csv")
    args = parser.parse_args()
    run(args.src, args.out)
