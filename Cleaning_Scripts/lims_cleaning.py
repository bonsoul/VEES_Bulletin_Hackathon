"""
lims_cleaning_improved.py

Cleans the raw LIMS "Diseases" combined dataset and writes a cleaned CSV
ready for loading into VEES_DATABASE.

Improvements over the original notebook:
  - Wrapped into functions (testable, re-runnable, no out-of-order cell bugs)
  - Category columns are filled with "Unknown" BEFORE being cast to
    category dtype, so the fill actually takes effect (in the notebook,
    category columns were cast to category *before* the fillna step, so
    fillna("Unknown") silently skipped them because "Unknown" wasn't yet
    a valid category)
  - Column drop happens using a single source of truth list, and no longer
    references `data1` before it exists (the notebook called
    `data1.columns` in the cell before `data1` was created)
  - Logging instead of bare prints, so you get a clear before/after summary
  - Config values (paths, column lists) live at the top of the file
  - Basic validation: warns if an expected column is missing instead of
    throwing a raw KeyError
  - CLI args so you can point it at different input/output files without
    editing the script

Usage:
    python lims_cleaning_improved.py \\
        --input "../Raw_Datasets/Diseases_LIMS_Combined.csv" \\
        --output "../Cleaned Data/lims_cleaned.csv"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lims_cleaning")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "Raw_Datasets" / "Diseases_LIMS_Combined.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Cleaned Data" / "lims_cleaned.csv"

# --------------------------------------------------------------------------
# Config — column groupings, edit here if the source schema changes
# --------------------------------------------------------------------------

DATE_COLS = [
    "DATE RECEIVED",
    "SAMPLING DATE",
    "RECEIVED IN SECTION",
    "RESULT DATE",
    "DATE REPORTED",
]

NULLABLE_INT_COLS = ["PRG.UNIT SAMPLE"]
INT_COLS = ["SUBMISSION NUMBER", "NUM SAMPLES TESTED"]

STRING_COLS = [
    "NOTIFIED TO", "ACCEPTING LAB", "TESTING LAB", "TESTING SECTION CODE",
    "TESTING SECTION", "SAMPLING PURPOSE", "PLAN", "COUNTY", "SUB-COUNTY",
    "WARD", "SPECIES", "SAMPLE TYPE", "SAMPLE IDENTIFICATION", "SEX", "AGE",
    "TEST", "METHOD", "SPECIFIC TEST", "SOP", "DISEASE", "RESULT SENTENCE",
    "SECOND RESULT SENTENCE", "INTERPRETATION", "source_file",
    "SUBMITTER", "SAMPLING POINT", "OWNER", "TESTED BY", "VALIDATED BY",
]

CATEGORY_COLS = [
    "ACCEPTING LAB", "TESTING LAB", "TESTING SECTION CODE", "TESTING SECTION",
    "SAMPLING PURPOSE", "PLAN", "COUNTY", "SUB-COUNTY", "WARD", "SPECIES",
    "SAMPLE TYPE", "SEX", "TEST", "METHOD", "DISEASE",
]

# Columns dropped from the final, analysis-ready table
COLS_TO_DROP = [
    "RECEIVED IN SECTION", "source_file", "TESTING SECTION CODE",
    "ACCEPTING LAB", "TESTING LAB", "RESULT DATE", "DATE RECEIVED",
    "SAMPLING DATE", "DATE REPORTED",
]

SEX_MAP = {"M": "Male", "F": "Female", "S": "Unknown", "N": "Unknown"}


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    log.info("Loading raw data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded shape: %s", df.shape)
    return df


def _existing(cols: list[str], df: pd.DataFrame, label: str) -> list[str]:
    """Return only the columns that are actually present, warning about any that aren't."""
    present = [c for c in cols if c in df.columns]
    missing = sorted(set(cols) - set(present))
    if missing:
        log.warning("%s: %d expected column(s) not found in data, skipping: %s",
                    label, len(missing), missing)
    return present


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Converting dtypes")

    for col in _existing(DATE_COLS, df, "date columns"):
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

    for col in _existing(NULLABLE_INT_COLS, df, "nullable int columns"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in _existing(INT_COLS, df, "int columns"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    for col in _existing(STRING_COLS, df, "string columns"):
        df[col] = df[col].astype("string")

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Handling missing values")

    # Drop columns that are entirely null — nothing to clean there
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]
    if dropped:
        log.info("Dropped %d fully-empty column(s)", dropped)

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].fillna("Unknown")

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to category dtype AFTER filling missing values, so 'Unknown'
    is a real observed value rather than being silently skipped."""
    log.info("Applying category dtypes")
    for col in _existing(CATEGORY_COLS, df, "category columns"):
        df[col] = df[col].astype("category")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = _existing(COLS_TO_DROP, df, "columns to drop")
    log.info("Dropping %d irrelevant column(s): %s", len(cols), cols)
    return df.drop(columns=cols)


def recode_sex(df: pd.DataFrame) -> pd.DataFrame:
    if "SEX" not in df.columns:
        log.warning("SEX column not found, skipping recode")
        return df
    log.info("Recoding SEX values")
    df["SEX"] = df["SEX"].astype("string").replace(SEX_MAP).fillna("Unknown")
    return df


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "Null Count": df.isnull().sum(),
        "Null Percentage": (df.isnull().sum() / len(df) * 100).round(2),
    }).sort_values(by="Null Percentage", ascending=False)
    return summary


def clean(input_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = convert_dtypes(df)

    log.info("Null summary BEFORE filling:\n%s", null_summary(df))

    df = handle_missing(df)
    df = apply_categories(df)
    df = drop_irrelevant_columns(df)
    df = recode_sex(df)

    log.info("Null summary AFTER cleaning:\n%s", null_summary(df))
    log.info("Final shape: %s", df.shape)
    return df


def save_cleaned(df: pd.DataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Saved cleaned data to %s", out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the LIMS diseases dataset.")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to the raw combined CSV.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path to write the cleaned CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean(args.input)
    save_cleaned(df, args.output)


if __name__ == "__main__":
    main()