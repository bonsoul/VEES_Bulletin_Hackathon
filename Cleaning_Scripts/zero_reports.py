"""Clean the KABS zero-report dataset and save it as CSV."""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "Raw_Datasets" / "Zero Report_KABS.xls"
OUTPUT_PATH = PROJECT_ROOT / "Cleaned Data" / "zero_cleaned.csv"

COLUMNS_TO_DROP = [
    "Submitter UUID",
    "Submitter Username",
    "Submitter Name",
    "Submitter Organization",
    "Submitter Role",
    "Id",
    "Date of Report",
    "Location.1",
    "Location",
    "Other Species",
    "Notes",
]


def clean_zero_reports(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and clean the zero-report workbook."""
    zero_report = pd.read_excel(input_path)

    # Remove metadata and fields that are not needed for analysis.
    zero_clean_df = zero_report.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Fill numeric fields, including latitude and longitude, with zero.
    numeric_columns = zero_clean_df.select_dtypes(include="number").columns
    zero_clean_df[numeric_columns] = zero_clean_df[numeric_columns].fillna(0)

    # Add and use "Unknown" for missing categorical values.
    categorical_columns = zero_clean_df.select_dtypes(include="category").columns
    for column in categorical_columns:
        if "Unknown" not in zero_clean_df[column].cat.categories:
            zero_clean_df[column] = zero_clean_df[column].cat.add_categories(["Unknown"])
        zero_clean_df[column] = zero_clean_df[column].fillna("Unknown")

    # Fill missing text values with a readable placeholder.
    string_columns = zero_clean_df.select_dtypes(include=["string", "object"]).columns
    zero_clean_df[string_columns] = zero_clean_df[string_columns].fillna("Unknown")

    return zero_clean_df


def main() -> None:
    cleaned_df = clean_zero_reports()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(cleaned_df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
