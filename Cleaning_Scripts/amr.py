"""Clean the active AMR surveillance dataset and save it as CSV."""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = (
    PROJECT_ROOT / "Raw_Datasets" / "ACTIVE AMR SURVEILLANCE DATA_NVLs.xlsx"
)
OUTPUT_PATH = PROJECT_ROOT / "Cleaned Data" / "AMR_CLEANED.csv"

COLUMNS_TO_DROP = [
    "ACCEPTING_LAB",
    "MEASUREMENT_UNIT",
    "NUM_SAMPLES_TESTED",
    "TESTING_LAB",
    "TESTING_SECTION_CODE",
    "TESTING_SECTION",
    "ORIGINAL_MATERIAL",
    "DATE_REPORTED",
    "OLD_SUBMISSION_NUMBER",
    "PRG.UNIT_SAMPLE",
    "TERRITORY_1",
    "TERRITORY_2",
    "TERRITORY_3",
    "SAMPLE_IDENTIFICATION",
    "OLD_SAMPLE_IDENTIFICATION",
    "RESULT_1",
    "RESULT_2",
    "DATE_RECEIVED",
    "SUBMISSION_NUMBER",
    "TEST",
    "METHOD",
    "RESULT_DATE",
]


def clean_amr_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and clean the AMR surveillance dataset."""
    amr_df = pd.read_excel(input_path)

    # Remove laboratory and submission metadata not needed for analysis.
    cleaned_df = amr_df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    return cleaned_df


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    cleaned_df = clean_amr_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(cleaned_df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
