"""
Load cleaned VEES events data into PostgreSQL.

Project structure (updated):

VEES_Bulletin_Hackathon/
│
├── Cleaned Data/
│   └── Kabs_cleaned.csv
│
└── Postgre_Schemes/
    ├── load_to_postgres.py
    ├── schema.sql
    └── .env                (create this yourself — see below)

Setup (one time):

    pip install sqlalchemy psycopg2-binary pandas

Database credentials default to postgres/1234/localhost/5432/VEES_DATABASE
below. To use different credentials without editing this file, set
environment variables before running (PowerShell):

    $env:PGUSER = "postgres"
    $env:PGPASSWORD = "yourpassword"
    $env:PGHOST = "localhost"
    $env:PGPORT = "5432"
    $env:PGDATABASE = "VEES_DATABASE"

Run from the project root:

    python Postgre_Schemes\\load_to_postgres.py

Or specify a different input file (CSV or parquet — detected by extension):

    python Postgre_Schemes\\load_to_postgres.py --input "Cleaned Data/Kabs_cleaned.csv"
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
log = logging.getLogger("vees_loader")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_FILE = PROJECT_ROOT / "Postgre_Schemes" / "schema.sql"

# Updated default: points at the actual cleaned CSV location.
DEFAULT_INPUT = PROJECT_ROOT / "Cleaned Data" / "Kabs_cleaned.csv"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Local defaults for this hackathon project. Override any of these by
# setting the matching environment variable before running the script,
# e.g. in PowerShell:  $env:PGPASSWORD = "something_else"
DEFAULTS = {
    "PGUSER": "postgres",
    "PGPASSWORD": "1234",
    "PGHOST": "localhost",
    "PGPORT": "5432",
    "PGDATABASE": "VEES_DATABASE",
}


def load_config() -> dict:
    """
    Build database credentials: environment variables win if set,
    otherwise fall back to the local defaults above.
    """

    config = {key: os.environ.get(key, default) for key, default in DEFAULTS.items()}

    log.info("Using database config (PGPASSWORD hidden): "
              "user=%s host=%s port=%s db=%s",
              config["PGUSER"], config["PGHOST"], config["PGPORT"], config["PGDATABASE"])

    return config


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
# READ INPUT FILE (CSV or Parquet)
# ============================================================

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Read the cleaned data file into a pandas DataFrame.
    Supports .csv and .parquet, detected from the file extension.
    """

    log.info("Checking input file...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found at:\n{input_path}")

    log.info("Input file: %s", input_path)

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        log.info("Reading CSV file...")
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        log.info("Reading parquet file...")
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Expected .csv or .parquet."
        )

    if df.empty:
        raise ValueError("The input file contains no rows.")

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

    # Support both the cleaned schema and the notebook's original headers.
    column_aliases = {
        "Id": "event_id",
        "Date of Report": "report_date",
        "County": "county",
        "Sub-County": "sub_county",
        "Ward": "ward",
        "Locality": "locality",
        "Species Affected": "species_affected",
        "Number at Risk": "number_at_risk",
        "Number Sick / Bitten": "number_sick_bitten",
        "Number Dead": "number_dead",
        "Disease/Condition": "disease_condition",
        "Nature of Diagnosis": "nature_of_diagnosis",
        "Number of Humans Affected (If zoonosis)": "humans_affected",
        "Disease Control Method": "control_methods",
        "Longitude": "longitude",
        "Latitude": "latitude",
        "Number Sick": "number_sick",
    }
    df = df.rename(columns=column_aliases)

    if "event_id" not in df.columns:
        log.warning(
            "No event_id column found; generating stable IDs from each row."
        )
        row_values = df.astype("string").fillna("").agg("|".join, axis=1)
        df.insert(
            0,
            "event_id",
            row_values.map(
                lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            ),
        )

    # Any list/array-valued cells (e.g. multi-select columns) break
    # SQL parameter binding — convert them to comma-joined strings so the
    # upsert doesn't blow up on 'unhashable type: list'.
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list,))).any():
            log.info("Converting list-valued column '%s' to string.", col)
            df[col] = df[col].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
            )

    if "control_methods" in df.columns:
        def parse_control_methods(value: object) -> list[str]:
            if isinstance(value, list):
                return [str(item) for item in value]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text_value = str(value).strip()
            if text_value.startswith("[") and text_value.endswith("]"):
                text_value = text_value[1:-1]
            if not text_value:
                return []
            return [item.strip().strip("'\"") for item in text_value.split(",")]

        df["control_methods"] = df["control_methods"].apply(parse_control_methods)

    # Pandas NaN doesn't play well with psycopg2 param binding for
    # non-numeric columns — normalize to None so NULLs load cleanly.
    df = df.where(pd.notnull(df), None)

    duplicate_ids = df["event_id"].duplicated().sum()

    if duplicate_ids > 0:
        log.warning(
            "%s duplicate event_id values found in the input file.",
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
                records = [
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
        description="Load cleaned VEES events data (CSV or Parquet) into PostgreSQL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the cleaned CSV or parquet file. "
             "Defaults to 'Cleaned Data/Kabs_cleaned.csv'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows per upsert batch (default: 5000)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    input_path = input_path.resolve()

    config = load_config()

    log.info("=" * 65)
    log.info("VEES → PostgreSQL DATA LOADER")
    log.info("=" * 65)
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Input file   : %s", input_path)
    log.info("Schema       : %s", SCHEMA_FILE)
    log.info("Database     : %s", config["PGDATABASE"])
    log.info("Host         : %s:%s", config["PGHOST"], config["PGPORT"])
    log.info("=" * 65)

    try:
        engine = get_engine(config)
        test_connection(engine)
        apply_schema(engine)

        df = load_data(input_path)
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