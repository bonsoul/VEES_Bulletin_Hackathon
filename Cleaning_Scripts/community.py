from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "Raw_Datasets" / "Community Reports_ KABS.csv"
OUTPUT_PATH = PROJECT_ROOT / "Cleaned Data" / "community_data.csv"


def clean_community_data() -> pd.DataFrame:
    """Load and clean the community reports dataset."""
    comm_df = pd.read_csv(INPUT_PATH)

    # Remove submitter metadata and fields not needed for analysis.
    columns_to_drop = [
        "Submitter UUID",
        "Other sign",
        "Other sign.1",
        "Other Sign",
        "Other Species",
        "Specify Other Sign",
        "Specify Other Sign.1",
        "Submitter Username",
        "Submitter Name",
        "Submitter Organization",
        "Submitter Role",
        "Id",
        "Date of Start of sickness",
        "Location",
        "Location.1",
        "Unnamed: 27",
    ]
    comm_df1 = comm_df.drop(columns=columns_to_drop)

    # Convert each field to the type needed for analysis.
    comm_df1["Date of Report"] = pd.to_datetime(
        comm_df1["Date of Report"], errors="coerce"
    )
    # Invalid coordinate values become missing before they are filled below.
    comm_df1["Longitude"] = pd.to_numeric(
        comm_df1["Longitude"], errors="coerce"
    ).astype("float64")
    comm_df1["Latitude"] = pd.to_numeric(
        comm_df1["Latitude"], errors="coerce"
    ).astype("float64")

    count_columns = [
        "Total Number of Animals in the herd",
        "Number Sick",
        "Number Dead",
    ]
    # Nullable integers preserve invalid or missing counts until they are handled.
    comm_df1[count_columns] = comm_df1[count_columns].apply(
        pd.to_numeric, errors="coerce"
    ).astype("Int64")

    text_columns = [
        "County",
        "Sub-County",
        "Ward",
        "Village",
        "Animals Affected",
        "Signs of Disease",
    ]
    comm_df1[text_columns] = comm_df1[text_columns].astype("string")

    # Use zero for missing coordinates and Unknown for missing villages.
    comm_df1["Longitude"] = comm_df1["Longitude"].fillna(0.0)
    comm_df1["Latitude"] = comm_df1["Latitude"].fillna(0.0)
    comm_df1["Village"] = comm_df1["Village"].fillna("Unknown")

    return comm_df1


def main() -> None:
    comm_df1 = clean_community_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comm_df1.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
