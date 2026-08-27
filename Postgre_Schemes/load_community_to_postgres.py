"""Load the cleaned community reports dataset into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Cleaned Data" / "community_data.csv"
DB_SCHEMA = "vees"
DB_TABLE = "community_reports"


SOURCE_COLUMNS = {
    "Date of Report": "report_date",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "County": "county",
    "Sub-County": "sub_county",
    "Ward": "ward",
    "Village": "village",
    "Animals Affected": "animals_affected",
    "Signs of Disease": "signs_of_disease",
    "Total Number of Animals in the herd": "total_animals_in_herd",
    "Number Sick": "number_sick",
    "Number Dead": "number_dead",
}


def get_engine():
    database_url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:"
        f"{os.getenv('PGPASSWORD', '1234')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}"
        f"/{os.getenv('PGDATABASE', 'VEES_DATABASE')}"
    )
    return create_engine(database_url, pool_pre_ping=True)


def create_table(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}";
            CREATE TABLE IF NOT EXISTS "{DB_SCHEMA}"."{DB_TABLE}" (
                event_id              TEXT PRIMARY KEY,
                report_date           DATE,
                longitude             DOUBLE PRECISION NOT NULL DEFAULT 0,
                latitude              DOUBLE PRECISION NOT NULL DEFAULT 0,
                county                TEXT,
                sub_county            TEXT,
                ward                  TEXT,
                village               TEXT NOT NULL DEFAULT 'Unknown',
                animals_affected      TEXT,
                signs_of_disease      TEXT,
                total_animals_in_herd INTEGER,
                number_sick           INTEGER,
                number_dead           INTEGER,
                loaded_at             TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))


def read_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = set(SOURCE_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")
    if df.empty:
        raise ValueError("The community CSV contains no rows.")

    df = df[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(0.0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").fillna(0.0)
    count_columns = ["total_animals_in_herd", "number_sick", "number_dead"]
    df[count_columns] = df[count_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    df["village"] = df["village"].fillna("Unknown")

    # Create a stable key so rerunning the loader updates existing rows.
    df.insert(0, "event_id", [
        hashlib.sha256(
            "|".join("" if pd.isna(value) else str(value) for value in row)
            .encode("utf-8")
        ).hexdigest()
        for row in df.itertuples(index=False, name=None)
    ])
    return df.astype(object).where(pd.notna(df), None)


def load_data(engine, df: pd.DataFrame, batch_size: int) -> None:
    columns = list(df.columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    update_columns = [column for column in columns if column != "event_id"]
    update_clause = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )
    statement = text(f"""
        INSERT INTO "{DB_SCHEMA}"."{DB_TABLE}" ({quoted_columns})
        VALUES ({placeholders})
        ON CONFLICT (event_id) DO UPDATE SET {update_clause}
    """)

    with engine.begin() as connection:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            connection.execute(statement, batch.to_dict(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    df = read_data(input_path.resolve())
    engine = get_engine()

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    create_table(engine)
    load_data(engine, df, args.batch_size)
    print(f"Loaded {len(df):,} community reports into {DB_SCHEMA}.{DB_TABLE}.")


if __name__ == "__main__":
    main()
