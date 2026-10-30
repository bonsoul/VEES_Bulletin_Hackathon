"""Clean and validate the KABS vaccination export without modifying the source."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "kabs2.0_raw" / "vf_29102026.csv"
DEFAULT_OUTPUT = ROOT / "Cleaned Data" / "vf_29102026_clean.csv"
DEFAULT_REJECTS = ROOT / "Cleaned Data" / "vf_29102026_rejected.csv"
DEFAULT_DUPLICATES = ROOT / "Cleaned Data" / "vf_29102026_duplicates.csv"

REQUIRED_COLUMNS = [
    "county",
    "subcounty",
    "ward",
    "vaccination_site",
    "gps_location",
    "vaccination_date",
    "species",
    "disease",
    "total_number_vaccinated",
    "vaccination_type",
    "batch_number",
    "number_of_beneficiaries",
    "other_disease",
]
REQUIRED_VALUES = [
    "county",
    "subcounty",
    "ward",
    "vaccination_date",
    "species",
    "disease",
    "total_number_vaccinated",
    "number_of_beneficiaries",
]
OPTIONAL_TEXT = [
    "vaccination_site",
    "vaccination_type",
    "batch_number",
]
NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "nat"}
KENYA_BOUNDS = {"min_lat": -5.1, "max_lat": 5.2, "min_lon": 33.5, "max_lon": 42.2}

DISEASE_ALIASES = {
    "fowl pox": "Fowl pox",
    "newcastle disease (nd)": "Newcastle Disease",
    "east coast fever": "East Coast Fever (ECF)",
    "infectious bural disease (gumboro)": "Infectious Bursal Disease (Gumboro)",
    "infectious bursai disease (gumboro)": "Infectious Bursal Disease (Gumboro)",
    "contagious calpine pleuropneumonia (ccpp)": (
        "Contagious Caprine Pleuropneumonia (CCPP)"
    ),
    "contagious capine pleuropneumonia (ccpp)": (
        "Contagious Caprine Pleuropneumonia (CCPP)"
    ),
}
VACCINATION_TYPE_ALIASES = {
    "routine vaccination": "Routine vaccination",
    "mass vaccination": "Mass vaccination",
    "outbreak/ring vaccination": "Outbreak/Ring Vaccination",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("clean_vf")


def normalize_text(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.str.casefold().isin(NULL_TOKENS), pd.NA)


def normalize_gps(value: object) -> tuple[str, str | None]:
    """Return canonical GPS text and an error, if the supplied point is invalid."""
    if pd.isna(value):
        return "Not recorded", None

    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value).strip(),
    )
    if not match:
        return "Not recorded", "GPS is not in [latitude, longitude] format"

    latitude, longitude = map(float, match.groups())
    if not (
        KENYA_BOUNDS["min_lat"] <= latitude <= KENYA_BOUNDS["max_lat"]
        and KENYA_BOUNDS["min_lon"] <= longitude <= KENYA_BOUNDS["max_lon"]
    ):
        return "Not recorded", "GPS point is outside the configured Kenya bounds"
    return f"[{match.group(1)}, {match.group(2)}]", None


def clean_data(
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cleaned, rejected, and duplicate records."""
    df = pd.read_csv(input_path, dtype="string", keep_default_na=False)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError(f"No records found in {input_path}")

    df = df[REQUIRED_COLUMNS].copy()
    df.insert(0, "source_row", range(2, len(df) + 2))
    for column in REQUIRED_COLUMNS:
        df[column] = normalize_text(df[column])

    # Normalize known disease and vaccination-type spelling/case variants.
    df["disease"] = df["disease"].map(
        lambda value: DISEASE_ALIASES.get(value.casefold(), value)
        if pd.notna(value)
        else value
    )
    df["vaccination_type"] = df["vaccination_type"].map(
        lambda value: VACCINATION_TYPE_ALIASES.get(value.casefold(), value)
        if pd.notna(value)
        else value
    )

    date_text = df["vaccination_date"].str.replace(
        r"\s+(?:EAT|East Africa Time)$", "", regex=True
    )
    parsed_dates = pd.to_datetime(
        date_text, format="%a, %b %d, %Y", errors="coerce"
    )
    df["vaccination_date"] = parsed_dates.dt.strftime("%Y-%m-%d").astype("string")

    numeric_columns = [
        "total_number_vaccinated",
        "number_of_beneficiaries",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rejected_reasons: dict[int, list[str]] = {}
    for column in REQUIRED_VALUES:
        missing_mask = df[column].isna()
        for index in df.index[missing_mask]:
            rejected_reasons.setdefault(index, []).append(
                f"missing or invalid required field: {column}"
            )
    for column in numeric_columns:
        negative_mask = df[column].lt(0).fillna(False)
        fractional_mask = df[column].notna() & df[column].mod(1).ne(0)
        for index in df.index[negative_mask]:
            rejected_reasons.setdefault(index, []).append(
                f"negative value in {column}"
            )
        for index in df.index[fractional_mask]:
            rejected_reasons.setdefault(index, []).append(
                f"non-integer value in {column}"
            )

    invalid_rows = sorted(rejected_reasons)
    rejected = df.loc[invalid_rows].copy()
    if not rejected.empty:
        rejected.insert(
            1,
            "rejection_reason",
            ["; ".join(rejected_reasons[index]) for index in invalid_rows],
        )

    valid = df.drop(index=invalid_rows).copy()
    valid["total_number_vaccinated"] = valid[
        "total_number_vaccinated"
    ].astype("int64")
    valid["number_of_beneficiaries"] = valid[
        "number_of_beneficiaries"
    ].astype("int64")

    invalid_gps: list[tuple[int, str]] = []
    gps_values = []
    for row in valid[["source_row", "gps_location"]].itertuples(index=False):
        gps, error = normalize_gps(row.gps_location)
        gps_values.append(gps)
        if error:
            invalid_gps.append((row.source_row, error))
    valid["gps_location"] = gps_values
    if invalid_gps:
        log.warning(
            "%d GPS values were missing or invalid and are marked 'Not recorded'.",
            len(invalid_gps),
        )

    for column in OPTIONAL_TEXT:
        valid[column] = valid[column].fillna("Not reported")
    valid["other_disease"] = valid["other_disease"].fillna(
        "Not reported"
    )
    not_applicable = valid["disease"].str.casefold().ne("other")
    valid.loc[not_applicable & valid["other_disease"].eq("Not reported"), "other_disease"] = (
        "Not applicable"
    )

    # No source identifier is present; remove only identical full records.
    duplicate_mask = valid[REQUIRED_COLUMNS].duplicated(keep="first")
    duplicates = valid.loc[duplicate_mask].copy()
    if not duplicates.empty:
        duplicates.insert(1, "duplicate_reason", "identical normalized record")
    cleaned = valid.loc[~duplicate_mask].copy()
    cleaned = cleaned[REQUIRED_COLUMNS]

    validate_cleaned(cleaned)
    log.info(
        "Read %d rows; kept %d unique rows, rejected %d incomplete/invalid rows, "
        "and recorded %d exact duplicates.",
        len(df),
        len(cleaned),
        len(rejected),
        len(duplicates),
    )
    return cleaned, rejected, duplicates


def validate_cleaned(df: pd.DataFrame) -> None:
    """Fail loudly if the output violates the expected clean-data contract."""
    if df.empty:
        raise ValueError("Cleaning produced no valid records.")
    if df.isna().any().any():
        columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Cleaned output still has null values in: {columns}")
    if df[REQUIRED_COLUMNS].duplicated().any():
        raise ValueError("Cleaned output still contains duplicate records.")
    if (df["total_number_vaccinated"] < 0).any():
        raise ValueError("Cleaned output contains negative vaccination totals.")
    if (df["number_of_beneficiaries"] < 0).any():
        raise ValueError("Cleaned output contains negative beneficiary counts.")
    if not df["vaccination_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Cleaned output contains a non-ISO vaccination date.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTS)
    parser.add_argument("--duplicates", type=Path, default=DEFAULT_DUPLICATES)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    cleaned, rejected, duplicates = clean_data(args.input)
    for output_path, frame in (
        (args.output, cleaned),
        (args.rejected, rejected),
        (args.duplicates, duplicates),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        log.info("Saved %d rows to %s", len(frame), output_path)


if __name__ == "__main__":
    main()
