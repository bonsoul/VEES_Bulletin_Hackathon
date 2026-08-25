"""
load_to_postgres.py

Loads the cleaned LIMS CSV (output of lims_cleaning_improved.py) into
PostgreSQL, database VEES_DATABASE.

Credentials:
    By default this reads the connection string from the DATABASE_URL
    environment variable so you don't hardcode a password in the script.
    You can also pass --db-url explicitly if you'd rather match the
    %sql postgresql://postgres:1234@localhost:5432/VEES_DATABASE
    connection you were using in Jupyter, e.g.:

        python load_to_postgres.py --db-url "postgresql://postgres:1234@localhost:5432/VEES_DATABASE"

Usage:
    python load_to_postgres.py \\
        --csv "../Cleaned Data/lims_cleaned.csv" \\
        --table lims_diseases \\
        --if-exists replace
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("load_to_postgres")

DEFAULT_DB_URL = "postgresql://postgres:1234@localhost:5432/VEES_DATABASE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "Cleaned Data" / "lims_cleaned.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned LIMS CSV into Postgres.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                         help="Path to the cleaned CSV to load.")
    parser.add_argument("--table", default="lims_diseases",
                         help="Destination table name.")
    parser.add_argument("--schema", default="vees",
                         help="Destination schema (default: vees).")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
        help="SQLAlchemy connection string. Defaults to the DATABASE_URL "
             "env var, falling back to the VEES_DATABASE connection.",
    )
    parser.add_argument(
        "--if-exists", choices=["fail", "replace", "append"], default="replace",
        help="Behaviour if the table already exists (default: replace).",
    )
    parser.add_argument("--chunksize", type=int, default=5000,
                         help="Rows per batch insert (default: 5000).")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    log.info("Reading cleaned CSV from %s", path)
    # parse_dates left off here on purpose: any *_DATE / DATE_* columns are
    # picked up automatically below so this works even if the column list
    # in the cleaning script changes.
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded shape: %s", df.shape)

    date_like = [c for c in df.columns if "DATE" in c.upper()]
    for col in date_like:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if date_like:
        log.info("Parsed date-like columns: %s", date_like)

    # Postgres wants lowercase, underscore-separated identifiers, not
    # "SUB-COUNTY" / "PRG.UNIT SAMPLE" style headers.
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def get_engine(db_url: str):
    log.info("Connecting to database")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Connection OK")
    return engine


def load_to_db(df: pd.DataFrame, engine, table: str, schema: str,
                if_exists: str, chunksize: int) -> None:
    log.info(
        "Writing %d rows into %s.%s (if_exists=%s)",
        len(df), schema, table, if_exists,
    )
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    with engine.connect() as conn:
        count = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')).scalar()
    log.info("Load complete. %s.%s now has %d row(s)", schema, table, count)


def main() -> None:
    args = parse_args()

    if args.db_url == DEFAULT_DB_URL:
        log.warning(
            "Using the default hardcoded connection string. Prefer setting "
            "the DATABASE_URL environment variable or passing --db-url so "
            "the password isn't sitting in the script/shell history."
        )

    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    df = load_csv(csv_path)
    engine = get_engine(args.db_url)

    try:
        load_to_db(df, engine, args.table, args.schema, args.if_exists, args.chunksize)
    except Exception:
        log.exception("Load failed")
        sys.exit(1)


if __name__ == "__main__":
    main()