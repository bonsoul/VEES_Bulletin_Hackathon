"""Load cleaned KABS zero reports into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Cleaned Data" / "zero_cleaned.csv"
DB_SCHEMA = "vees"
DB_TABLE = "zero_reports"


def get_engine():
    database_url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{os.getenv('PGPASSWORD', '1234')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}"
        f"/{os.getenv('PGDATABASE', 'VEES_DATABASE')}"
    )
    return create_engine(database_url, pool_pre_ping=True)


def create_table(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};
            CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
                event_id TEXT PRIMARY KEY,
                longitude DOUBLE PRECISION,
                latitude DOUBLE PRECISION,
                county TEXT NOT NULL,
                sub_county TEXT NOT NULL,
                ward TEXT NOT NULL,
                locality TEXT NOT NULL,
                species_examined TEXT NOT NULL,
                production_system TEXT NOT NULL,
                rvf TEXT,
                rp TEXT,
                ppr TEXT,
                fmd TEXT,
                ccpp TEXT,
                cbpp TEXT,
                avian_influenza TEXT,
                loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))


def read_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {
        "Longitude", "Latitude", "County", "Sub-County", "Ward",
        "Locality", "Species examined", "Production System", "RVF",
        "RP", "PPR", "FMD", "CCPP", "CBPP", "Avian Influenza",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    df = df.rename(columns={
        "Sub-County": "sub_county",
        "Species examined": "species_examined",
        "Production System": "production_system",
        "Avian Influenza": "avian_influenza",
        "CCPP": "ccpp",
    })
    df.columns = [column.lower() for column in df.columns]
    df = df.where(pd.notna(df), None)
    df.insert(0, "event_id", [
        hashlib.sha256(
            "|".join("" if value is None else str(value) for value in row).encode()
        ).hexdigest()
        for row in df.itertuples(index=False, name=None)
    ])
    return df


def load_data(engine, df: pd.DataFrame, batch_size: int = 1000) -> None:
    columns = list(df.columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    update_columns = [column for column in columns if column != "event_id"]
    update_clause = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )
    statement = text(f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} ({quoted_columns})
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
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    df = read_data(input_path)
    create_table(engine)
    load_data(engine, df, args.batch_size)
    print(f"Loaded {len(df):,} zero reports into {DB_SCHEMA}.{DB_TABLE}.")


if __name__ == "__main__":
    main()