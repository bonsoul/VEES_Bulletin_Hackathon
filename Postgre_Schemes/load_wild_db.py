"""Load the cleaned wildlife dataset into PostgreSQL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "Cleaned Data" / "wildlife_cleaned_df.csv"
DEFAULT_DB_URL = "postgresql://postgres:1234@localhost:5432/VEES_DATABASE"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Load the cleaned wildlife CSV into PostgreSQL."
	)
	parser.add_argument(
		"--csv",
		type=Path,
		default=DEFAULT_CSV,
		help="Path to the cleaned wildlife CSV.",
	)
	parser.add_argument("--schema", default="vees")
	parser.add_argument("--table", default="wildlife_cleaned")
	parser.add_argument(
		"--if-exists",
		choices=["fail", "replace", "append"],
		default="replace",
	)
	parser.add_argument("--chunksize", type=int, default=1000)
	parser.add_argument(
		"--db-url",
		default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
		help="PostgreSQL connection URL. Defaults to DATABASE_URL or the local database.",
	)
	return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Wildlife CSV was not found at: {path}")

	df = pd.read_csv(path, low_memory=False)
	if df.empty:
		raise ValueError("The wildlife CSV contains no rows.")

	# PostgreSQL identifiers cannot safely use spaces, punctuation, or duplicates.
	normalized_columns = (
		df.columns.astype(str)
		.str.strip()
		.str.lower()
		.str.replace(r"[^0-9a-z]+", "_", regex=True)
		.str.strip("_")
	)
	df.columns = normalized_columns

	if df.columns.duplicated().any():
		raise ValueError("The CSV contains duplicate column names after normalization.")

	return df


def load_to_postgres(
	df: pd.DataFrame,
	db_url: str,
	schema: str,
	table: str,
	if_exists: str,
	chunksize: int,
) -> int:
	engine = create_engine(db_url, pool_pre_ping=True)
	with engine.begin() as connection:
		connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

	df.to_sql(
		name=table,
		con=engine,
		schema=schema,
		if_exists=if_exists,
		index=False,
		chunksize=chunksize,
		method="multi",
	)

	with engine.connect() as connection:
		count = connection.execute(
			text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
		).scalar_one()
	return int(count)


def main() -> None:
	args = parse_args()
	csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
	df = load_csv(csv_path.resolve())
	count = load_to_postgres(
		df=df,
		db_url=args.db_url,
		schema=args.schema,
		table=args.table,
		if_exists=args.if_exists,
		chunksize=args.chunksize,
	)
	print(f"Loaded {count:,} rows into {args.schema}.{args.table}")


if __name__ == "__main__":
	main()
