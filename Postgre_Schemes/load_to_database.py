"""
VEES PostgreSQL Data Loader

Loads a cleaned Parquet dataset into:

    Database: VEES_DATABASE
    Schema:   vees
    Table:    events_clean

Usage:

    python Postgre_Schemes\\load_to_postgres.py

Or specify the exact Parquet file:

    python Postgre_Schemes\\load_to_postgres.py --parquet "path\\to\\file.parquet"
"""

from pathlib import Path
import argparse
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = PROJECT_ROOT / "Postgre_Schemes" / "schema.sql"

# PostgreSQL settings
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "VEES_DATABASE"

# PostgreSQL destination
DB_SCHEMA = "vees"
DB_TABLE = "events_clean"

# Number of rows loaded per batch
BATCH_SIZE = 5000


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_engine():

    database_url = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True
    )


# ============================================================
# TEST DATABASE CONNECTION
# ============================================================

def test_connection(engine):

    print()
    print("Connecting to PostgreSQL...")

    try:

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("✓ PostgreSQL connection successful.")

    except SQLAlchemyError as e:

        print("✗ Could not connect to PostgreSQL.")
        print()
        print(e)

        sys.exit(1)


# ============================================================
# FIND PARQUET FILE
# ============================================================

def find_parquet():

    print()
    print("Searching for Parquet files...")

    parquet_files = list(
        PROJECT_ROOT.rglob("*.parquet")
    )

    if not parquet_files:

        print()
        print("✗ No Parquet files were found.")
        print()
        print("Your project currently contains no .parquet file.")
        print()
        print("Place your cleaned Parquet file somewhere inside:")
        print(PROJECT_ROOT)
        print()
        print("Then run the script again.")

        sys.exit(1)

    if len(parquet_files) == 1:

        print(
            f"✓ Found Parquet file:\n"
            f"  {parquet_files[0]}"
        )

        return parquet_files[0]

    print()
    print("Multiple Parquet files were found:")

    for i, file in enumerate(parquet_files, start=1):
        print(f"  [{i}] {file}")

    print()

    while True:

        choice = input(
            "Select the file number to load: "
        )

        try:

            choice = int(choice)

            if 1 <= choice <= len(parquet_files):
                return parquet_files[choice - 1]

        except ValueError:
            pass

        print("Invalid selection. Try again.")


# ============================================================
# APPLY DATABASE SCHEMA
# ============================================================

def apply_schema(engine):

    print()
    print("Checking schema file...")

    if not SCHEMA_FILE.exists():

        print(
            f"✗ schema.sql not found:\n"
            f"{SCHEMA_FILE}"
        )

        sys.exit(1)

    print(f"Schema file: {SCHEMA_FILE}")

    sql = SCHEMA_FILE.read_text(
        encoding="utf-8"
    )

    if not sql.strip():

        print("✗ schema.sql is empty.")
        sys.exit(1)

    print("Applying database schema...")

    try:

        with engine.begin() as connection:
            connection.execute(text(sql))

        print("✓ Database schema applied successfully.")

    except SQLAlchemyError as e:

        print("✗ Failed to apply schema.")
        print()
        print(e)

        sys.exit(1)


# ============================================================
# READ PARQUET
# ============================================================

def read_parquet(parquet_file):

    print()
    print("Reading Parquet file:")
    print(parquet_file)

    if not parquet_file.exists():

        print("✗ Parquet file does not exist.")
        sys.exit(1)

    try:

        df = pd.read_parquet(
            parquet_file
        )

    except Exception as e:

        print("✗ Could not read Parquet file.")
        print()
        print(e)

        sys.exit(1)

    if df.empty:

        print("✗ Parquet file contains no rows.")
        sys.exit(1)

    print()
    print("✓ Parquet loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df):

    print()
    print("Validating dataset...")

    # Convert column names to strings
    df.columns = df.columns.astype(str)

    # Check event_id
    if "event_id" not in df.columns:

        print()
        print("✗ ERROR: 'event_id' column not found.")
        print()
        print("Available columns:")

        for column in df.columns:
            print(f"  - {column}")

        sys.exit(1)

    # Remove duplicate event IDs
    duplicates = df["event_id"].duplicated().sum()

    if duplicates > 0:

        print(
            f"⚠ Found {duplicates:,} duplicate event_id values."
        )

        df = df.drop_duplicates(
            subset=["event_id"],
            keep="last"
        )

        print(
            f"✓ Duplicates removed."
            f" {len(df):,} rows remaining."
        )

    # Convert list values to strings
    for column in df.columns:

        contains_lists = df[column].apply(
            lambda x: isinstance(x, list)
        ).any()

        if contains_lists:

            print(
                f"Converting list values in '{column}' "
                f"to strings..."
            )

            df[column] = df[column].apply(
                lambda x: ", ".join(map(str, x))
                if isinstance(x, list)
                else x
            )

    print("✓ Dataset validation completed.")

    return df


# ============================================================
# LOAD DATA INTO POSTGRESQL
# ============================================================

def load_data(engine, df):

    print()
    print(
        f"Loading data into "
        f"{DB_SCHEMA}.{DB_TABLE}..."
    )

    columns = list(df.columns)

    quoted_columns = ", ".join(
        f'"{column}"'
        for column in columns
    )

    placeholders = ", ".join(
        f":{column}"
        for column in columns
    )

    # Update all columns except event_id
    update_columns = [
        column
        for column in columns
        if column != "event_id"
    ]

    update_clause = ", ".join(
        f'"{column}" = EXCLUDED."{column}"'
        for column in update_columns
    )

    sql = text(
        f"""
        INSERT INTO "{DB_SCHEMA}"."{DB_TABLE}"
        ({quoted_columns})
        VALUES ({placeholders})

        ON CONFLICT (event_id)
        DO UPDATE SET
        {update_clause}
        """
    )

    total_rows = len(df)

    try:

        with engine.begin() as connection:

            for start in range(
                0,
                total_rows,
                BATCH_SIZE
            ):

                end = min(
                    start + BATCH_SIZE,
                    total_rows
                )

                batch = df.iloc[
                    start:end
                ]

                records = batch.to_dict(
                    orient="records"
                )

                connection.execute(
                    sql,
                    records
                )

                print(
                    f"  Loaded {end:,} / "
                    f"{total_rows:,} rows"
                )

        print()
        print(
            f"✓ Successfully loaded "
            f"{total_rows:,} rows."
        )

    except SQLAlchemyError as e:

        print()
        print("✗ Failed to load data.")
        print()
        print(e)

        sys.exit(1)


# ============================================================
# VERIFY TABLE
# ============================================================

def verify_table(engine):

    print()
    print("Verifying PostgreSQL table...")

    query = text(
        f"""
        SELECT COUNT(*)
        FROM "{DB_SCHEMA}"."{DB_TABLE}";
        """
    )

    try:

        with engine.connect() as connection:

            count = connection.execute(
                query
            ).scalar()

        print()
        print(
            f"✓ {DB_SCHEMA}.{DB_TABLE} "
            f"contains {count:,} rows."
        )

    except SQLAlchemyError as e:

        print("⚠ Could not verify table.")
        print(e)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Load VEES cleaned Parquet "
            "data into PostgreSQL."
        )
    )

    parser.add_argument(
        "--parquet",
        type=Path,
        help=(
            "Full or relative path to "
            "the Parquet file."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("VEES → PostgreSQL DATA LOADER")
    print("=" * 65)

    print(f"Project : {PROJECT_ROOT}")
    print(f"Database: {DB_NAME}")
    print(f"Host    : {DB_HOST}:{DB_PORT}")
    print(f"Schema  : {DB_SCHEMA}")
    print(f"Table   : {DB_TABLE}")

    print("=" * 65)

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    engine = get_engine()

    test_connection(engine)

    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    apply_schema(engine)

    # --------------------------------------------------------
    # PARQUET
    # --------------------------------------------------------

    if args.parquet:

        parquet_file = args.parquet

        if not parquet_file.is_absolute():

            parquet_file = (
                PROJECT_ROOT / parquet_file
            )

        parquet_file = parquet_file.resolve()

    else:

        parquet_file = find_parquet()

    # --------------------------------------------------------
    # READ DATA
    # --------------------------------------------------------

    df = read_parquet(
        parquet_file
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    df = validate_data(df)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    load_data(
        engine,
        df
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    verify_table(
        engine
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("✓ DATA LOAD COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()