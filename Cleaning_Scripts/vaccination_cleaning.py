"""
Vaccination Data Cleaning Script
Cleans raw vaccination report data by dropping unnecessary columns,
handling missing values appropriately, and saving to CSV.
"""

import numpy as np
import pandas as pd


def clean_vaccination_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Clean the raw vaccination data.

    Parameters
    ----------
    input_path : str
        Path to the raw Excel file.
    output_path : str
        Path to save the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    # Load the data
    df = pd.read_excel(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Drop inappropriate / sensitive columns
    drop_cols = [
        "Id",
        "Latitude.1",
        "Longitude.1",
        "Submitter Name",
        "Submitter Role",
        "Submitter Username",
        "Submitter UUID",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"Dropped columns: {drop_cols}")

    # Numeric columns: keep as NaN (proper missing value)
    num_cols = ["Latitude", "Longitude"]
    df[num_cols] = df[num_cols].fillna(np.nan)

    # String columns: fill with "Unknown"
    str_cols = ["Specify other disease", "Vaccination Site(s)", "Submitter Organization"]
    df[str_cols] = df[str_cols].fillna("Unknown")

    # Ensure correct data types
    if "Total number vaccinated" in df.columns:
        df["Total number vaccinated"] = pd.to_numeric(df["Total number vaccinated"], errors="coerce")
    if "Number of beneficiaries (HHs)" in df.columns:
        df["Number of beneficiaries (HHs)"] = pd.to_numeric(df["Number of beneficiaries (HHs)"], errors="coerce")

    print("\nCleaned data info:")
    print(df.info())

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")

    return df


if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\ADMIN\Documents\VEES_Bulletin_Hackathon\Datasets\reports_2026-08-18.xls"
    OUTPUT_FILE = r"C:\Users\ADMIN\Documents\VEES_Bulletin_Hackathon\Datasets\vaccination_cleaned.csv"

    cleaned_df = clean_vaccination_data(INPUT_FILE, OUTPUT_FILE)
