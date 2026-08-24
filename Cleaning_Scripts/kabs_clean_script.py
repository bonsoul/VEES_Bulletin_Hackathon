"""
KABS livestock disease event data cleaning pipeline.
Reads the raw ND1 form export, cleans it, and writes a silver-layer
parquet ready for loading into PostgreSQL.

Run: python clean_events.py --raw data/raw/combined_reports.xlsx --out data/silver/events_clean.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kabs_clean")

# Columns confirmed unused for the silver layer. "Id" is kept — it's the
# only natural primary key in the raw export; the original script dropped it.
COLS_TO_DROP = [
    "Date Event Reported", "Test Used",
    "Number Affected", *[f"Number Affected.{i}" for i in range(1, 11)],
    "Death", "Location", "Location.1",
    "Submitter UUID", "Submitter Username", "Submitter Name",
    "Source Form Version", "Submitter Organization", "Submitter Role",
    "Hemorrhagic Signs", "Neurologic signs", "Animal Bites",
    "Respiratory Signs", "Oral/Foot Lesions", "Cutaneous/Skin Lesions",
    "Gastrointestinal tract syndromes", "Other Syndromes",
    "Number Destroyed", "Number Vaccinated",
    "Date of Start of Outbreak/Event", "Production System",
    "Number Slaughtered", "Abortion", "Sudden Death",
]

OTHER_MAP = {
    "Disease/Condition": "Other Disease",
    "Disease Control Method": "Other Control Method",
    "Species Affected": "Other Species",
}

NUM_FILL_ZERO = ["Number Sick / Bitten", "Number Sick", "Number of Humans Affected (If zoonosis)"]
TEXT_COLS = ["County", "Sub-County", "Ward", "Locality", "Disease/Condition", "Species Affected"]

RENAME = {
    "Id": "event_id",
    "Date of Report": "report_date",
    "County": "county",
    "Sub-County": "sub_county",
    "Ward": "ward",
    "Locality": "locality",
    "Species Affected": "species_affected",
    "Number at Risk": "number_at_risk",
    "Number Sick / Bitten": "number_sick_bitten",
    "Number Dead": "number_dead",
    "Disease/Condition": "disease_condition",
    "Nature of Diagnosis": "nature_of_diagnosis",
    "Number of Humans Affected (If zoonosis)": "humans_affected",
    "Disease Control Method": "control_methods",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Number Sick": "number_sick",
}


def load_raw(path: Path) -> pd.DataFrame:
    log.info("Loading raw file: %s", path)
    df = pd.read_excel(path)
    before = len(df)
    df = df.drop_duplicates(subset="Id")
    if len(df) != before:
        log.warning("Dropped %d duplicate Id rows", before - len(df))
    log.info("Loaded %d rows, %d columns", *df.shape)
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    out = df.drop(columns=cols)
    log.info("Dropped %d columns -> %d remain", len(cols), out.shape[1])
    return out


def merge_other_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for main_col, other_col in OTHER_MAP.items():
        if other_col not in df.columns:
            continue
        df[main_col] = df[main_col].astype("string")
        df[other_col] = df[other_col].astype("string")
        mask = df[main_col].isna() | (df[main_col] == "Other")
        df.loc[mask, main_col] = df.loc[mask, other_col]
    return df.drop(columns=[c for c in OTHER_MAP.values() if c in df.columns])


def fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in NUM_FILL_ZERO if c in df.columns]
    df[cols] = df[cols].fillna(0)
    return df


def fill_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for group_col in ["Ward", "Sub-County", "County"]:
        for coord in ["Latitude", "Longitude"]:
            df[coord] = df[coord].fillna(df.groupby(group_col)[coord].transform("mean"))
    remaining = int(df[["Latitude", "Longitude"]].isna().sum().sum())
    if remaining:
        log.warning("%d coordinate values still missing after hierarchy fill", remaining)
    return df


def fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Nature of Diagnosis", "Disease/Condition"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")
    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()
    return df


def parse_control_method(df: pd.DataFrame) -> pd.DataFrame:
    """'[Treatment, Vaccination]' -> ['Treatment', 'Vaccination'] for a Postgres text[] column."""
    df = df.copy()
    col = "Disease Control Method"

    def to_list(s):
        if pd.isna(s):
            return []
        s = str(s).strip("[]")
        return [x.strip() for x in s.split(",") if x.strip()]

    df[col] = df[col].apply(to_list)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date of Report"] = pd.to_datetime(df["Date of Report"], errors="coerce")
    bad = int(df["Date of Report"].isna().sum())
    if bad:
        log.warning("%d rows have unparseable 'Date of Report'", bad)
    return df


def run_pipeline(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = drop_unused(df)
    df = merge_other_columns(df)
    df = fill_numeric(df)
    df = fill_coordinates(df)
    df = fill_categoricals(df)
    df = normalize_text(df)
    df = parse_control_method(df)
    df = parse_dates(df)
    df = df.rename(columns=RENAME)[list(RENAME.values())]

    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if not remaining_nulls.empty:
        log.warning("Columns with remaining nulls:\n%s", remaining_nulls)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/combined_reports.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("data/silver/events_clean.parquet"))
    args = parser.parse_args()
    run_pipeline(args.raw, args.out)
