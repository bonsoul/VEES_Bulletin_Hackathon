"""Load cleaned AMR surveillance data into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "Cleaned Data" / "AMR_CLEANED.csv"
DB_SCHEMA = "vees"
DB_TABLE = "amr_clean"

EXPECTED_COLUMNS = [
    "SAMPLING_PURPOSE",
    "SAMPLING_DATE",
    "ORIGINAL_SPECIES",
    "BREED",
    "ANIMAL_TYPE",
    "AGE_CATEGORY",
    "SAMPLE_TYPE",
    "SEX",
    "AGE",
    "ORGANISM",
    "ANTIBIOTIC",
    "DISEASE",
    "INTERPRETATION",
]


def get_engine():
    """Create a PostgreSQL engine using environment variables or local defaults."""
    database_url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{os.getenv('PGPASSWORD', '1234')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}"
        f"/{os.getenv('PGDATABASE', 'VEES_DATABASE')}"
    )
    return create_engine(database_url, pool_pre_ping=True)


def create_table(engine) -> None:
    """Create the AMR destination table if it does not already exist."""
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};
            CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
                amr_id TEXT PRIMARY KEY,
                sampling_purpose TEXT,
                sampling_date DATE,
                original_species TEXT,
                breed TEXT,
                animal_type TEXT,
                age_category TEXT,
                sample_type TEXT,
                sex TEXT,
                age TEXT,
                organism TEXT,
                antibiotic TEXT,
                disease TEXT,
                interpretation TEXT,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))


def read_data(input_path: Path) -> pd.DataFrame:
    """Read and prepare the cleaned AMR CSV for database loading."""
    dataframe = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    dataframe.columns = dataframe.columns.str.strip()
    missing = set(EXPECTED_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    dataframe = dataframe[EXPECTED_COLUMNS].rename(columns={
        column: column.lower().strip().replace(" ", "_")
        for column in EXPECTED_COLUMNS
    })
    dataframe["sampling_date"] = pd.to_datetime(
        dataframe["sampling_date"], dayfirst=True, errors="coerce"
    ).dt.date
    dataframe = dataframe.replace({"": None, " ": None})

    # Include the row position so identical source records remain separate rows.
    dataframe.insert(0, "amr_id", [
        hashlib.sha256(f"{index}|{row}".encode()).hexdigest()
        for index, row in enumerate(dataframe.itertuples(index=False, name=None))
    ])
    return dataframe


def load_data(engine, dataframe: pd.DataFrame, batch_size: int) -> None:
    """Upsert AMR records in batches."""
    columns = list(dataframe.columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    update_clause = ", ".join(
        f'"{column}" = EXCLUDED."{column}"'
        for column in columns
        if column != "amr_id"
    )
    statement = text(f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} ({quoted_columns})
        VALUES ({placeholders})
        ON CONFLICT (amr_id) DO UPDATE SET {update_clause}
    """)

    with engine.begin() as connection:
        for start in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[start:start + batch_size]
            connection.execute(statement, batch.to_dict(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataframe = read_data(input_path)
    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    create_table(engine)
    load_data(engine, dataframe, args.batch_size)
    print(f"Loaded {len(dataframe):,} AMR records into {DB_SCHEMA}.{DB_TABLE}.")


if __name__ == "__main__":
    main()
