# VEES Data Cleaning and Database

Cleaning pipelines and PostgreSQL loaders for the VEES Bulletin datasets.

## Project Structure

- `Raw_Datasets/`: Original Excel and CSV files.
- `Cleaning_Scripts/`: Python and notebook-based cleaning workflows.
- `Cleaned Data/`: Generated CSV files used for analysis and database loading.
- `Postgre_Schemes/`: PostgreSQL schema and dataset-specific loaders.

## Requirements

- Python 3.13 or later
- PostgreSQL running locally
- PostgreSQL database named `VEES_DATABASE`

Create or use the project virtual environment and install the Python dependencies:

```powershell
cd "C:\Users\ADMIN\Documents\VEES_Bulletin_Hackathon"
\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`openpyxl` and `xlrd` provide support for reading `.xlsx` and `.xls` files.

## Cleaning Scripts

Run these commands from the project root:

```powershell
\.venv\Scripts\python.exe Cleaning_Scripts\amr.py
\.venv\Scripts\python.exe Cleaning_Scripts\wildlife_cleaning.py
\.venv\Scripts\python.exe Cleaning_Scripts\zero_reports.py
```

Generated files:

| Dataset | Output |
| --- | --- |
| AMR surveillance | `Cleaned Data/AMR_CLEANED.csv` |
| KABS events | `Cleaned Data/Kabs_cleaned.csv` |
| LIMS disease records | `Cleaned Data/lims_cleaned.csv` |
| Vaccination records | `Cleaned Data/vaccination_cleaned.csv` |
| Wildlife events | `Cleaned Data/wildlife_cleaned_df.csv` |
| KABS zero reports | `Cleaned Data/zero_cleaned.csv` |

The AMR cleaning script removes laboratory and submission metadata. The wildlife cleaning script converts date and numeric fields and fills missing values. The zero-report cleaning script removes submitter metadata and fills missing values with `Unknown` or `0` where appropriate.

## PowerShell Commands

Run these commands from the project root. Generate the cleaned files first:

```powershell
cd "C:\Users\ADMIN\Documents\VEES_Bulletin_Hackathon"
\.venv\Scripts\python.exe Cleaning_Scripts\amr.py
\.venv\Scripts\python.exe Cleaning_Scripts\wildlife_cleaning.py
\.venv\Scripts\python.exe Cleaning_Scripts\zero_reports.py
```

Set PostgreSQL connection values for the current PowerShell session:

```powershell
$env:PGUSER = "postgres"
$env:PGPASSWORD = "your_password"
$env:PGHOST = "localhost"
$env:PGPORT = "5432"
$env:PGDATABASE = "VEES_DATABASE"
```

Install the database dependencies and run the loaders:

```powershell
\.venv\Scripts\python.exe -m pip install sqlalchemy psycopg2-binary
\.venv\Scripts\python.exe Postgre_Schemes\load_amr.py
\.venv\Scripts\python.exe Postgre_Schemes\load_zero_reports.py
\.venv\Scripts\python.exe Postgre_Schemes\load_to_postgres.py --input "Cleaned Data\Kabs_cleaned.csv"
```

## PostgreSQL Loading

The loaders default to these local connection settings when environment variables are not set:

```text
User:     postgres
Password: 1234
Host:     localhost
Port:     5432
Database: VEES_DATABASE
```

Run the AMR loader:

```powershell
\.venv\Scripts\python.exe Postgre_Schemes\load_amr.py
```

This reads `Cleaned Data/AMR_CLEANED.csv`, creates `vees.amr_clean`, and loads 1,475 AMR records using repeatable upserts.

Run the zero-report loader:

```powershell
\.venv\Scripts\python.exe Postgre_Schemes\load_zero_reports.py
```

This reads `Cleaned Data/zero_cleaned.csv`, creates `vees.zero_reports`, and loads the zero-report records using repeatable upserts.

The general events loader is also available:

```powershell
\.venv\Scripts\python.exe Postgre_Schemes\load_to_postgres.py
```

It loads `Cleaned Data/Kabs_cleaned.csv` into `vees.events_clean` using `Postgre_Schemes/schema.sql`.

The same variables are used by all three loaders. Each loader creates its destination table if it does not exist and uses an upsert so it can be run again safely.

## SQL Setup and Verification

Create the database once from a PostgreSQL client such as `psql`:

```sql
CREATE DATABASE "VEES_DATABASE";
```

Connect to `VEES_DATABASE` and apply the shared events schema:

```powershell
psql -U postgres -h localhost -d VEES_DATABASE -f Postgre_Schemes\schema.sql
```

The dataset loaders also create their required tables automatically. The current database tables are:

| Dataset | PostgreSQL table |
| --- | --- |
| KABS events | `vees.events_clean` |
| AMR surveillance | `vees.amr_clean` |
| KABS zero reports | `vees.zero_reports` |

Verify row counts and inspect sample records with SQL:

```sql
SELECT COUNT(*) FROM vees.amr_clean;
SELECT COUNT(*) FROM vees.zero_reports;
SELECT COUNT(*) FROM vees.events_clean;

SELECT * FROM vees.amr_clean LIMIT 5;
SELECT * FROM vees.zero_reports LIMIT 5;
SELECT * FROM vees.events_clean LIMIT 5;
```

The cleaned LIMS, vaccination, and wildlife CSV files are available in `Cleaned Data/`. Dedicated PostgreSQL loaders for those datasets have not yet been added.
