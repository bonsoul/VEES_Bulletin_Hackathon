"""
Load cleaned VEES vaccination data into PostgreSQL.

Input   : Datasets/vaccination cleaned.csv   (output of vaccination_cleaning.py)
Target  : vees.vaccination

NOTE ON THE PRIMARY KEY:
The cleaned vaccination CSV has no unique id column (the original "Id"
column is dropped during cleaning). To make repeated loads safe
(re-running this script shouldn't create duplicate rows), this script
generates a deterministic "vaccination_id" for each row by hashing a
set of columns that together identify a unique vaccination record:
County, Sub county, Ward, Vaccination Date, Species, Disease, and
Vaccination Site(s). Two rows with identical values in all of these
columns will collapse into one row on re-load (upsert), so if your
data can legitimately contain exact duplicates of all those fields,
let me know and I'll adjust the key.

Setup (one time):

    pip install sqlalchemy psycopg2-binary pandas

Database credentials default to postgres/1234/localhost/5432/VEES_DATABASE
below. Override via environment variables before running (PowerShell):

    $env:PGUSER = "postgres"
    $env:PGPASSWORD = "yourpassword"
    $env:PGHOST = "localhost"
    $env:PGPORT = "5432"
    $env:PGDATABASE = "VEES_DATABASE"

Run from the project root:

    python Postgres_DB\\load_vaccination_to_db.py

Or specify a different input file:

    python Postgres_DB\\load_vaccination_to_db.py --input "Datasets\\vaccination cleaned.csv"
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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
log = logging.getLogger("vaccination_loader")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_FILE = SCRIPT_DIR / "schema.sql"

DEFAULT_INPUT = PROJECT_ROOT / "Cleaned Data" / "vaccination_cleaned.csv"

# Columns used to build a deterministic surrogate key, since the
# cleaned data has no natural unique id. See module docstring.
KEY_COLUMNS = [
    "County",
    "Sub county",
    "Ward",
    "Vaccination Date",
    "Species",
    "Disease",
    "Vaccination Site(s)",
]

TABLE_NAME = "vaccination"
SCHEMA_NAME = "vees"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DEFAULTS = {
    "PGUSER": "postgres",
    "PGPASSWORD": "1234",
    "PGHOST": "localhost",
    "PGPORT": "5432",
    "PGDATABASE": "VEES_DATABASE",
}


def load_config() -> dict:
    config = {key: os.environ.get(key, default) for key, default in DEFAULTS.items()}
    log.info("Using database config (PGPASSWORD hidden): "
              "user=%s host=%s port=%s db=%s",
              config["PGUSER"], config["PGHOST"], config["PGPORT"], config["PGDATABASE"])
    return config


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_engine(config: dict) -> Engine:
    database_url = (
        f"postgresql+psycopg2://{config['PGUSER']}:{config['PGPASSWORD']}"
        f"@{config['PGHOST']}:{config['PGPORT']}/{config['PGDATABASE']}"
    )
    return create_engine(database_url, pool_pre_ping=True)


def test_connection(engine: Engine) -> None:
    log.info("Connecting to PostgreSQL...")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        log.info("✓ PostgreSQL connection successful.")
    except SQLAlchemyError:
        log.exception("✗ PostgreSQL connection failed.")
        raise


# ============================================================
# SCHEMA / TABLE
# ============================================================

def ensure_table(engine: Engine) -> None:
    """
    Create the vaccination table if it doesn't already exist.
    Column types are kept permissive (TEXT / NUMERIC) since this
    script doesn't assume a pre-existing schema.sql definition for
    this table — adjust if you already have one.
    """

    log.info("Ensuring target table exists (%s.%s)...", SCHEMA_NAME, TABLE_NAME)

    ddl = text(f"""
        CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}";

        CREATE TABLE IF NOT EXISTS "{SCHEMA_NAME}"."{TABLE_NAME}" (
            vaccination_id            TEXT PRIMARY KEY,
            "Submitter Organization"  TEXT,
            "County"                  TEXT,
            "Sub county"              TEXT,
            "Ward"                    TEXT,
            "Vaccination Site(s)"     TEXT,
            "Latitude"                DOUBLE PRECISION,
            "Longitude"               DOUBLE PRECISION,
            "Vaccination Date"        TEXT,
            "Species"                 TEXT,
            "Disease"                 TEXT,
            "Specify other disease"   TEXT,
            "Total number vaccinated" INTEGER,
            "Number of beneficiaries (HHs)" INTEGER
        );
    """)

    try:
        with engine.begin() as connection:
            connection.execute(ddl)
        log.info("✓ Table ready.")
    except SQLAlchemyError:
        log.exception("✗ Failed to ensure table exists.")
        raise


# ============================================================
# READ INPUT
# ============================================================

def load_data(input_path: Path) -> pd.DataFrame:
    log.info("Checking input file...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found at:\n{input_path}")

    log.info("Input file: %s", input_path)
    log.info("Reading CSV file...")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The input file contains no rows.")

    log.info("✓ Loaded %s rows, %s columns.", f"{len(df):,}", f"{len(df.columns):,}")

    return df


# ============================================================
# VALIDATE / KEY GENERATION
# ============================================================

def build_surrogate_key(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build a unique key — missing expected column(s): {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    def make_key(row) -> str:
        raw = "|".join(str(row[c]) for c in KEY_COLUMNS)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    df = df.copy()
    df["vaccination_id"] = df.apply(make_key, axis=1)

    # Put the key column first for readability.
    cols = ["vaccination_id"] + [c for c in df.columns if c != "vaccination_id"]
    df = df[cols]

    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Validating data...")

    df.columns = df.columns.astype(str)

    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, list)).any():
            log.info("Converting list-valued column '%s' to string.", col)
            df[col] = df[col].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
            )

    df = df.where(pd.notnull(df), None)

    duplicate_ids = df["vaccination_id"].duplicated().sum()
    if duplicate_ids > 0:
        log.warning(
            "%s duplicate vaccination_id values found (rows with identical "
            "County/Sub county/Ward/Vaccination Date/Species/Disease/Site).",
            f"{duplicate_ids:,}",
        )
        df = df.drop_duplicates(subset=["vaccination_id"], keep="last")
        log.info("✓ Removed duplicates. %s rows remaining.", f"{len(df):,}")

    log.info("✓ Data validation completed.")

    return df


# ============================================================
# UPSERT
# ============================================================

def upsert(engine: Engine, df: pd.DataFrame, batch_size: int = 5000) -> None:
    if df.empty:
        log.info("No data to insert.")
        return

    columns = list(df.columns)
    quoted_columns = ", ".join(f'"{c}"' for c in columns)
    parameter_names = {column: f"value_{index}" for index, column in enumerate(columns)}
    value_placeholders = ", ".join(f":{parameter_names[c]}" for c in columns)
    update_columns = [c for c in columns if c != "vaccination_id"]

    update_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_columns)
    conflict_clause = f"ON CONFLICT (vaccination_id) DO UPDATE SET {update_clause}"

    insert_sql = text(f"""
        INSERT INTO "{SCHEMA_NAME}"."{TABLE_NAME}" ({quoted_columns})
        VALUES ({value_placeholders})
        {conflict_clause}
    """)

    total = len(df)
    log.info("Loading %s records into %s.%s (batch size %s)...",
              f"{total:,}", SCHEMA_NAME, TABLE_NAME, batch_size)

    # Parameter binding requires column names without special characters
    # like spaces or parentheses to map cleanly via to_dict — SQLAlchemy
    # handles this fine since we bind by the same names used in :placeholders,
    # but dict keys must match exactly, so keep df column names as-is.
    try:
        with engine.begin() as connection:
            for start in range(0, total, batch_size):
                batch = df.iloc[start:start + batch_size]
                records = [
                    {parameter_names[c]: row[c] for c in columns}
                    for row in batch.to_dict(orient="records")
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

def check_table(engine: Engine) -> None:
    query = text(f'SELECT COUNT(*) FROM "{SCHEMA_NAME}"."{TABLE_NAME}";')
    try:
        with engine.connect() as connection:
            count = connection.execute(query).scalar()
        log.info("✓ %s.%s currently contains %s rows.", SCHEMA_NAME, TABLE_NAME, f"{count:,}")
    except SQLAlchemyError:
        log.warning("Could not check table row count.", exc_info=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load cleaned VEES vaccination data (CSV) into PostgreSQL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Path to the cleaned vaccination CSV.")
    parser.add_argument("--batch-size", type=int, default=5000,
                         help="Number of rows per upsert batch (default: 5000)")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    input_path = input_path.resolve()

    config = load_config()

    log.info("=" * 65)
    log.info("VEES VACCINATION → PostgreSQL DATA LOADER")
    log.info("=" * 65)
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Input file   : %s", input_path)
    log.info("Database     : %s", config["PGDATABASE"])
    log.info("Host         : %s:%s", config["PGHOST"], config["PGPORT"])
    log.info("=" * 65)

    try:
        engine = get_engine(config)
        test_connection(engine)
        ensure_table(engine)

        df = load_data(input_path)
        df = build_surrogate_key(df)
        df = validate_dataframe(df)

        log.info("Data summary: %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")

        upsert(engine=engine, df=df, batch_size=args.batch_size)

        check_table(engine)

    except Exception:
        log.exception("Data load failed — see error above.")
        sys.exit(1)

    log.info("=" * 65)
    log.info("✓ DATA LOAD COMPLETED SUCCESSFULLY")
    log.info("=" * 65)


if __name__ == "__main__":
    main()