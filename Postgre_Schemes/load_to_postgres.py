"""
Load cleaned VEES events data into PostgreSQL.

Project structure:

VEES_Bulletin_Hackathon/
│
├── data/
│   └── silver/
│       └── events_clean.parquet
│
└── Postgre_Schemes/
    ├── load_to_postgres.py
    ├── schema.sql
    └── .env                (create this yourself — see below)

Setup (one time):

    pip install python-dotenv sqlalchemy psycopg2-binary pandas pyarrow

Create a file called `.env` next to this script with:

    PGUSER=postgres
    PGPASSWORD=1234
    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=VEES_DATABASE

Run from the project root:

    python Postgre_Schemes\\load_to_postgres.py

Or specify a different parquet file:

    python Postgre_Schemes\\load_to_postgres.py --parquet data/silver/events_clean.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vees_loader")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_FILE = SCRIPT_DIR / "schema.sql"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "silver" / "events_clean.parquet"
ENV_FILE = SCRIPT_DIR / ".env"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

def load_config() -> dict:
    """
    Load database credentials from environment variables / .env file.
    Never hardcode credentials in source — this keeps the script safe
    to commit to git.
    """

    # Load .env if it exists (does nothing if the file is absent)
    load_dotenv(ENV_FILE)

    required = ["PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGDATABASE"]
    missing = [key for key in required if not os.environ.get(key)]

    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        log.error(
            "Create a '.env' file at %s with these keys, "
            "or set them in your shell before running this script.",
            ENV_FILE,
        )
        sys.exit(1)

    return {key: os.environ[key] for key in required}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_engine(config: dict) -> Engine:
    """
    Create and return a SQLAlchemy PostgreSQL engine.
    """

    database_url = (
        f"postgresql+psycopg2://{config['PGUSER']}:{config['PGPASSWORD']}"
        f"@{config['PGHOST']}:{config['PGPORT']}/{config['PGDATABASE']}"
    )

    return create_engine(database_url, pool_pre_ping=True)


def test_connection(engine: Engine) -> None:
    """
    Test the PostgreSQL database connection.
    """

    log.info("Connecting to PostgreSQL...")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        log.info("✓ PostgreSQL connection successful.")

    except SQLAlchemyError:
        log.exception("✗ PostgreSQL connection failed.")
        raise


# ============================================================
# APPLY DATABASE SCHEMA
# ============================================================

def apply_schema(engine: Engine) -> None:
    """
    Execute schema.sql to create the required schema/table(s).
    """

    log.info("Checking schema file...")

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"schema.sql was not found at:\n{SCHEMA_FILE}")

    log.info("Schema file: %s", SCHEMA_FILE)

    sql = SCHEMA_FILE.read_text(encoding="utf-8")

    if not sql.strip():
        raise ValueError("schema.sql is empty.")

    log.info("Applying database schema...")

    try:
        with engine.begin() as connection:
            connection.execute(text(sql))
        log.info("✓ Database schema applied successfully.")

    except SQLAlchemyError:
        log.exception("✗ Failed to apply database schema.")
        raise


# ============================================================
# READ PARQUET
# ============================================================

def load_parquet(parquet_path: Path) -> pd.DataFrame:
    """
    Read the cleaned parquet file into a pandas DataFrame.
    """

    log.info("Checking parquet file...")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file was not found at:\n{parquet_path}")

    log.info("Parquet file: %s", parquet_path)
    log.info("Reading parquet file...")

    df = pd.read_parquet(parquet_path)

    if df.empty:
        raise ValueError("The parquet file contains no rows.")

    log.info("✓ Loaded %s rows, %s columns.", f"{len(df):,}", f"{len(df.columns):,}")

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic validation before loading data.
    """

    log.info("Validating data...")

    df.columns = df.columns.astype(str)

    if "event_id" not in df.columns:
        raise ValueError(
            "The dataframe does not contain an 'event_id' column.\n"
            "The upsert operation requires event_id."
        )

    # Any list/array-valued cells (e.g. multi-select columns) break
    # SQL parameter binding — convert them to tuples/strings so the
    # upsert doesn't blow up on 'unhashable type: list'.
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list,))).any():
            log.info("Converting list-valued column '%s' to string.", col)
            df[col] = df[col].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
            )

    duplicate_ids = df["event_id"].duplicated().sum()

    if duplicate_ids > 0:
        log.warning(
            "%s duplicate event_id values found in the parquet file.",
            f"{duplicate_ids:,}",
        )
        df = df.drop_duplicates(subset=["event_id"], keep="last")
        log.info("✓ Removed duplicates. %s rows remaining.", f"{len(df):,}")

    log.info("✓ Data validation completed.")

    return df


# ============================================================
# UPSERT DATA
# ============================================================

def upsert(
    engine: Engine,
    df: pd.DataFrame,
    table: str = "events_clean",
    schema: str = "vees",
    batch_size: int = 5000,
) -> None:
    """
    Insert records into PostgreSQL. Existing records with the same
    event_id are updated. Loads in batches to avoid memory/parameter
    limits on large files.
    """

    if df.empty:
        log.info("No data to insert.")
        return

    columns = list(df.columns)
    quoted_columns = ", ".join(f'"{c}"' for c in columns)
    value_placeholders = ", ".join(f":{c}" for c in columns)
    update_columns = [c for c in columns if c != "event_id"]

    if update_columns:
        update_clause = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in update_columns
        )
        conflict_clause = f"ON CONFLICT (event_id) DO UPDATE SET {update_clause}"
    else:
        conflict_clause = "ON CONFLICT (event_id) DO NOTHING"

    insert_sql = text(
        f"""
        INSERT INTO "{schema}"."{table}" ({quoted_columns})
        VALUES ({value_placeholders})
        {conflict_clause}
        """
    )

    total = len(df)
    log.info("Loading %s records into %s.%s (batch size %s)...",
              f"{total:,}", schema, table, batch_size)

    try:
        with engine.begin() as connection:
            for start in range(0, total, batch_size):
                batch = df.iloc[start:start + batch_size]
                records: list[dict[str, Any]] = [
                    {str(key): value for key, value in record.items()}
                    for record in batch.to_dict(orient="records")
                ]
                connection.execute(insert_sql, records)
                log.info("  ...%s / %s rows upserted",
                         f"{min(start + batch_size, total):,}", f"{total:,}")

        log.info("✓ Successfully upserted %s rows.", f"{total:,}")

    except SQLAlchemyError:
        log.exception("✗ Failed to load data into PostgreSQL.")
        raise


# ============================================================
# CHECK TABLE
# ============================================================

def check_table(engine: Engine, schema: str = "vees", table: str = "events_clean") -> None:
    """
    Check how many records currently exist in the table.
    """

    query = text(f'SELECT COUNT(*) FROM "{schema}"."{table}";')

    try:
        with engine.connect() as connection:
            count = connection.execute(query).scalar()
        log.info("✓ %s.%s currently contains %s rows.", schema, table, f"{count:,}")

    except SQLAlchemyError:
        log.warning("Could not check table row count.", exc_info=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Load cleaned VEES events data from Parquet into PostgreSQL."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to the cleaned parquet file. Defaults to data/silver/events_clean.parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows per upsert batch (default: 5000)",
    )
    args = parser.parse_args()

    parquet_path = args.parquet
    if not parquet_path.is_absolute():
        parquet_path = PROJECT_ROOT / parquet_path
    parquet_path = parquet_path.resolve()

    config = load_config()

    log.info("=" * 65)
    log.info("VEES → PostgreSQL DATA LOADER")
    log.info("=" * 65)
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Parquet      : %s", parquet_path)
    log.info("Schema       : %s", SCHEMA_FILE)
    log.info("Database     : %s", config["PGDATABASE"])
    log.info("Host         : %s:%s", config["PGHOST"], config["PGPORT"])
    log.info("=" * 65)

    try:
        engine = get_engine(config)
        test_connection(engine)
        apply_schema(engine)

        df = load_parquet(parquet_path)
        df = validate_dataframe(df)

        log.info("Data summary: %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")

        upsert(engine=engine, df=df, table="events_clean", schema="vees",
               batch_size=args.batch_size)

        check_table(engine)

    except Exception:
        log.exception("Data load failed — see error above.")
        sys.exit(1)

    log.info("=" * 65)
    log.info("✓ DATA LOAD COMPLETED SUCCESSFULLY")
    log.info("=" * 65)


if __name__ == "__main__":
    main()