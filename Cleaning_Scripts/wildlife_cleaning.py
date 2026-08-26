"""Clean the KABS wildlife-events dataset and save it as CSV."""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "Raw_Datasets" / "Wildlife events_KABS.xls"
OUTPUT_PATH = PROJECT_ROOT / "Cleaned Data" / "wildlife_cleaned_df.csv"

COLUMNS_TO_DROP = [
    "Location.1",
    "Date of rescue",
    "Other Area Type",
    "Bloody Diarrhoea",
    "Post Mortem(PM) done",
    "Evidence of bleeding",
    "Site Species rescued from",
    "Site Species rescued to",
    "Number of species rescued",
    "Date of Report",
    "Location",
    "Test Used",
    "Unnamed: 52",
    "Submitter Organization",
    "Unnamed: 68",
    "Submitter Role",
    "Submitter Username",
    "Submitter UUID",
    "Submitter Name",
    "Species rescued",
    "Number Affected.11",
    "Number Affected.10",
    "Number Affected.9",
    "Number Affected.8",
    "Number Affected.7",
    "Number Affected.6",
    "Number Affected.5",
    "Number Affected.4",
    "Number Affected.3",
    "Number Affected.2",
    "Number Affected.1",
    "PM findings",
    "Other Control Method",
    "Other Species",
]

DATE_COLUMNS = [
    "Date of Start of Outbreak/Event",
    "Date Event Reported",
]

NUMERIC_COLUMNS = [
    "Latitude",
    "Longitude",
    "Number Affected",
    "Number Sick / Bitten",
    "Number Dead",
    "Number of Humans Affected",
]


def clean_wildlife_events(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load, clean, and return the wildlife-events dataset."""
    wildlife_df = pd.read_excel(input_path)
    cleaned_df = wildlife_df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Convert only columns present in the input so the script handles schema changes.
    for column in DATE_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column], errors="coerce"
            )

    for column in NUMERIC_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column], errors="coerce"
            )

    # Use consistent pandas string values for remaining text columns.
    text_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns
    cleaned_df[text_columns] = cleaned_df[text_columns].astype("string")

    # Replace missing values with values suitable for database loading and analysis.
    numeric_columns = cleaned_df.select_dtypes(include="number").columns
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)

    categorical_columns = cleaned_df.select_dtypes(include="category").columns
    for column in categorical_columns:
        if "Unknown" not in cleaned_df[column].cat.categories:
            cleaned_df[column] = cleaned_df[column].cat.add_categories(["Unknown"])
        cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    string_columns = cleaned_df.select_dtypes(include=["string", "object"]).columns
    cleaned_df[string_columns] = cleaned_df[string_columns].fillna("Unknown")

    return cleaned_df


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    cleaned_df = clean_wildlife_events()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(cleaned_df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
