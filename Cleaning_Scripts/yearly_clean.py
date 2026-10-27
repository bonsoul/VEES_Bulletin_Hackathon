from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "Hackathon_Data" / "yearly_data.xls"

required_columns = ["Disease/Condition", "County", "No. Sick", "No. Dead"]
sheets = pd.read_excel(data_path, sheet_name=None)
cleaned_sheets = []

for sheet_name, sheet_df in sheets.items():
    sheet_df = sheet_df.rename(
        columns={
            "Number Sick": "No. Sick",
            "Number Sick / Bitten": "No. Sick",
            "Number Dead": "No. Dead",
        }
    )
    missing_columns = set(required_columns) - set(sheet_df.columns)
    if missing_columns:
        raise ValueError(
            f"Sheet {sheet_name!r} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    cleaned_sheets.append(sheet_df[required_columns].copy())

df_clean = pd.concat(cleaned_sheets, ignore_index=True)
output_path = project_root / "Hackathon_Data" / "yearly_cleaned.csv"
df_clean.to_csv(output_path, index=False)

print(f"Saved cleaned yearly data to: {output_path}")
print(f"Dataset shape: {df_clean.shape}")
print(df_clean.head())
